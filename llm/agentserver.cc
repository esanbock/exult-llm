/*
 *  agentserver.cc - Local TCP JSON-line server for the LLM agent bridge.
 *
 *  Copyright (C) 2026  The Exult Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

/*
 *  Threading model
 *  ---------------
 *  A dedicated NETWORK thread owns the listen/client sockets and does all
 *  blocking-ish socket work: accept, recv (splitting the byte stream into
 *  newline-delimited request lines), and send. It NEVER touches game state.
 *
 *  Request lines it parses out go into a mutex-guarded INBOUND queue.
 *
 *  The MAIN (game/render) thread calls Agent_server_poll() once per frame (and
 *  re-entrantly from the engine's blocking conversation loop). Poll drains the
 *  inbound queue and runs LLM_agent::handle_request()/begin_conversation() -
 *  which mutate live game state and MUST stay on the main thread - then pushes
 *  reply lines to a mutex-guarded OUTBOUND queue that the network thread sends.
 *
 *  This guarantees the agent's network I/O can never block the frame loop, and
 *  (with the A* node cap for gotos) that a single action can't hog it either.
 */

#ifdef HAVE_CONFIG_H
#	include <config.h>
#endif

#ifdef USE_LLM_AGENT

#include "agentserver.h"

#include "agent.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#	ifndef WIN32_LEAN_AND_MEAN
#		define WIN32_LEAN_AND_MEAN
#	endif
#	include <winsock2.h>
#	include <ws2tcpip.h>
using socket_t = SOCKET;
#	define AGENT_INVALID_SOCKET INVALID_SOCKET
#	define AGENT_CLOSESOCKET    closesocket
#else
#	include <arpa/inet.h>
#	include <fcntl.h>
#	include <netinet/in.h>
#	include <sys/socket.h>
#	include <unistd.h>
using socket_t = int;
#	define AGENT_INVALID_SOCKET (-1)
#	define AGENT_CLOSESOCKET    ::close
#endif

using std::cerr;
using std::endl;
using std::string;

namespace {

	constexpr int kDefaultPort = 45999;

	// Owned/used by the network thread only (except g_running/g_started which
	// are atomics read by the main thread).
	socket_t g_listen = AGENT_INVALID_SOCKET;
	socket_t g_client = AGENT_INVALID_SOCKET;
	string   g_inbuf;    // Partial-line accumulator (network thread only).
	bool     g_wsa = false;
	int      g_port = 0;

	std::thread       g_net_thread;
	std::atomic<bool> g_started{false};    // net thread should keep running
	std::atomic<bool> g_running{false};    // listen socket is up

	// Inbound: request lines parsed by the net thread, consumed by main thread.
	std::mutex             g_in_mtx;
	std::deque<string>     g_in_queue;

	// Outbound: reply lines produced by the main thread, sent by the net thread.
	std::mutex             g_out_mtx;
	std::deque<string>     g_out_queue;
	std::condition_variable g_out_cv;

	bool set_nonblocking(socket_t s) {
#ifdef _WIN32
		u_long mode = 1;
		return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
		int flags = fcntl(s, F_GETFL, 0);
		if (flags == -1) {
			return false;
		}
		return fcntl(s, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
	}

	bool would_block() {
#ifdef _WIN32
		return WSAGetLastError() == WSAEWOULDBLOCK;
#else
		return errno == EWOULDBLOCK || errno == EAGAIN;
#endif
	}

	// --- network thread helpers (run on the net thread only) -----------------

	void net_close_client() {
		if (g_client != AGENT_INVALID_SOCKET) {
			AGENT_CLOSESOCKET(g_client);
			g_client = AGENT_INVALID_SOCKET;
		}
		g_inbuf.clear();
		// Drop any queued state tied to the old client.
		{
			std::lock_guard<std::mutex> lk(g_in_mtx);
			g_in_queue.clear();
		}
		{
			std::lock_guard<std::mutex> lk(g_out_mtx);
			g_out_queue.clear();
		}
	}

	// Send all bytes of one line (+newline) on the client socket. Net thread.
	bool net_send_line(const string& s) {
		if (g_client == AGENT_INVALID_SOCKET) {
			return false;
		}
		string out = s;
		out += '\n';
		size_t sent = 0;
		while (sent < out.size()) {
			const int n = ::send(
					g_client, out.data() + sent,
					static_cast<int>(out.size() - sent), 0);
			if (n > 0) {
				sent += static_cast<size_t>(n);
			} else if (n < 0 && would_block()) {
				// brief spin; payloads are tiny
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			} else {
				return false;
			}
		}
		return true;
	}

	// Pull whatever is available on the client socket into g_inbuf and split
	// out complete lines into the inbound queue. Returns false if the client
	// disconnected/errored. Net thread.
	bool net_recv_lines() {
		char buf[2048];
		for (;;) {
			const int n = ::recv(g_client, buf, sizeof(buf), 0);
			if (n > 0) {
				g_inbuf.append(buf, static_cast<size_t>(n));
				if (g_inbuf.size() > (1u << 20)) {    // runaway client guard
					return false;
				}
				continue;
			}
			if (n == 0) {
				return false;    // orderly shutdown
			}
			if (would_block()) {
				break;    // nothing more right now
			}
			return false;    // real error
		}
		// Extract complete lines.
		size_t nl;
		while ((nl = g_inbuf.find('\n')) != string::npos) {
			string line = g_inbuf.substr(0, nl);
			g_inbuf.erase(0, nl + 1);
			if (!line.empty() && line.back() == '\r') {
				line.pop_back();
			}
			if (line.empty()) {
				continue;
			}
			std::lock_guard<std::mutex> lk(g_in_mtx);
			g_in_queue.push_back(std::move(line));
		}
		return true;
	}

	// Flush any pending outbound reply lines. Net thread.
	bool net_flush_out() {
		for (;;) {
			string line;
			{
				std::lock_guard<std::mutex> lk(g_out_mtx);
				if (g_out_queue.empty()) {
					return true;
				}
				line = std::move(g_out_queue.front());
				g_out_queue.pop_front();
			}
			if (!net_send_line(line)) {
				return false;
			}
		}
	}

	// The network thread main loop.
	void net_thread_main() {
		while (g_started.load()) {
			// Accept a client if we don't have one.
			if (g_client == AGENT_INVALID_SOCKET) {
				socket_t c = ::accept(g_listen, nullptr, nullptr);
				if (c != AGENT_INVALID_SOCKET) {
					set_nonblocking(c);
					g_client = c;
					g_inbuf.clear();
				} else {
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					continue;
				}
			}
			// Pump recv (parse lines -> inbound queue) and send (outbound).
			bool ok = net_recv_lines();
			if (ok) {
				ok = net_flush_out();
			}
			if (!ok) {
				net_close_client();
				continue;
			}
			// Light idle sleep; recv is non-blocking so avoid a busy spin.
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
		}
		net_close_client();
	}

}    // namespace

namespace LLM_agent {

	bool Agent_server_init(int port) {
		if (g_running.load()) {
			return true;    // already running
		}
		if (port <= 0) {
			port = kDefaultPort;
		}
		g_port = port;

#ifdef _WIN32
		if (!g_wsa) {
			WSADATA wsa;
			if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
				cerr << "LLM agent: WSAStartup failed" << endl;
				return false;
			}
			g_wsa = true;
		}
#endif

		g_listen = ::socket(AF_INET, SOCK_STREAM, 0);
		if (g_listen == AGENT_INVALID_SOCKET) {
			cerr << "LLM agent: socket() failed" << endl;
			return false;
		}

		int yes = 1;
		::setsockopt(
				g_listen, SOL_SOCKET, SO_REUSEADDR,
				reinterpret_cast<const char*>(&yes), sizeof(yes));

		sockaddr_in addr{};
		addr.sin_family      = AF_INET;
		addr.sin_port        = htons(static_cast<unsigned short>(port));
		addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);    // 127.0.0.1 only

		if (::bind(g_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
			cerr << "LLM agent: bind() to 127.0.0.1:" << port << " failed" << endl;
			AGENT_CLOSESOCKET(g_listen);
			g_listen = AGENT_INVALID_SOCKET;
			return false;
		}
		if (::listen(g_listen, 1) != 0) {
			cerr << "LLM agent: listen() failed" << endl;
			AGENT_CLOSESOCKET(g_listen);
			g_listen = AGENT_INVALID_SOCKET;
			return false;
		}
		set_nonblocking(g_listen);

		g_started.store(true);
		g_running.store(true);
		g_net_thread = std::thread(net_thread_main);
		cerr << "LLM agent: listening on 127.0.0.1:" << port
			 << " (network thread)" << endl;
		return true;
	}

	// Queue a reply line for the network thread to send.
	static void queue_reply(const string& s) {
		std::lock_guard<std::mutex> lk(g_out_mtx);
		g_out_queue.push_back(s);
	}

	// Pop one pending request line (produced by the net thread). Returns false
	// if none available.
	static bool pop_request(string& out) {
		std::lock_guard<std::mutex> lk(g_in_mtx);
		if (g_in_queue.empty()) {
			return false;
		}
		out = std::move(g_in_queue.front());
		g_in_queue.pop_front();
		return true;
	}

	void Agent_server_poll() {
		if (!g_running.load()) {
			return;
		}
		// Drain a SMALL number of requests per poll so agent action processing
		// (which runs here on the main thread) can never consume much of any
		// single frame. Excess requests wait in the queue for the next frame.
		// The network thread keeps receiving regardless, so nothing is lost.
		int budget = 2;
		string line;
		while (budget-- > 0 && pop_request(line)) {
			const string reply = LLM_agent::handle_request(line);
			if (reply.rfind("@TALK@", 0) == 0) {
				// Conversation: ack now so the client can send "answer" actions,
				// then run the (blocking) conversation. begin_conversation
				// re-enters Agent_server_poll() from the engine's answer loop;
				// because the network thread keeps filling the inbound queue,
				// those answer requests are drained and applied while blocked.
				const string name = reply.substr(6);
				queue_reply("{\"ok\":true,\"did\":\"talk\",\"starting\":true}");
				const bool ok = LLM_agent::begin_conversation(name);
				queue_reply(ok ? "{\"event\":\"conversation_ended\"}"
							   : "{\"event\":\"conversation_failed\"}");
			} else {
				queue_reply(reply);
			}
		}
	}

	void Agent_server_close() {
		g_started.store(false);
		if (g_net_thread.joinable()) {
			g_net_thread.join();
		}
		if (g_listen != AGENT_INVALID_SOCKET) {
			AGENT_CLOSESOCKET(g_listen);
			g_listen = AGENT_INVALID_SOCKET;
		}
		g_running.store(false);
		{
			std::lock_guard<std::mutex> lk(g_in_mtx);
			g_in_queue.clear();
		}
		{
			std::lock_guard<std::mutex> lk(g_out_mtx);
			g_out_queue.clear();
		}
#ifdef _WIN32
		if (g_wsa) {
			WSACleanup();
			g_wsa = false;
		}
#endif
	}

	bool Agent_server_running() {
		return g_running.load();
	}

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */
