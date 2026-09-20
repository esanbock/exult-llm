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

#ifdef HAVE_CONFIG_H
#	include <config.h>
#endif

#ifdef USE_LLM_AGENT

#include "agentserver.h"

#include "agent.h"

#include <cstring>
#include <iostream>
#include <string>

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

	socket_t g_listen = AGENT_INVALID_SOCKET;
	socket_t g_client = AGENT_INVALID_SOCKET;
	string   g_inbuf;    // Accumulates partial lines from the client.
	bool     g_wsa   = false;
	int      g_port  = 0;

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

	void close_client() {
		if (g_client != AGENT_INVALID_SOCKET) {
			AGENT_CLOSESOCKET(g_client);
			g_client = AGENT_INVALID_SOCKET;
		}
		g_inbuf.clear();
	}

	bool would_block() {
#ifdef _WIN32
		return WSAGetLastError() == WSAEWOULDBLOCK;
#else
		return errno == EWOULDBLOCK || errno == EAGAIN;
#endif
	}

	// Send all bytes of s on the client socket (best effort, blocking-ish;
	// payloads are tiny so this is fine).
	void send_line(const string& s) {
		if (g_client == AGENT_INVALID_SOCKET) {
			return;
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
				continue;    // retry
			} else {
				close_client();
				return;
			}
		}
	}

}    // namespace

namespace LLM_agent {

	bool Agent_server_init(int port) {
		if (g_listen != AGENT_INVALID_SOCKET) {
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
			Agent_server_close();
			return false;
		}
		if (::listen(g_listen, 1) != 0) {
			cerr << "LLM agent: listen() failed" << endl;
			Agent_server_close();
			return false;
		}
		set_nonblocking(g_listen);
		cerr << "LLM agent: listening on 127.0.0.1:" << port << endl;
		return true;
	}

	void Agent_server_poll() {
		if (g_listen == AGENT_INVALID_SOCKET) {
			return;
		}

		// Accept a new client if we don't have one.
		if (g_client == AGENT_INVALID_SOCKET) {
			socket_t c = ::accept(g_listen, nullptr, nullptr);
			if (c != AGENT_INVALID_SOCKET) {
				set_nonblocking(c);
				g_client = c;
				g_inbuf.clear();
			}
		}

		if (g_client == AGENT_INVALID_SOCKET) {
			return;
		}

		// Drain whatever is available without blocking.
		char buf[2048];
		for (;;) {
			const int n = ::recv(g_client, buf, sizeof(buf), 0);
			if (n > 0) {
				g_inbuf.append(buf, static_cast<size_t>(n));
				// Guard against unbounded growth from a misbehaving client.
				if (g_inbuf.size() > (1u << 20)) {
					g_inbuf.clear();
					close_client();
					return;
				}
				continue;
			}
			if (n == 0) {    // orderly shutdown
				close_client();
				return;
			}
			if (would_block()) {
				break;    // nothing more right now
			}
			close_client();    // real error
			return;
		}

		// Process complete lines.
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
			const string reply = LLM_agent::handle_request(line);
			if (reply.rfind("@TALK@", 0) == 0) {
				// Conversation request: ack immediately so the client can send
				// "answer" actions, then run the (blocking) conversation.  The
				// engine's answer loop polls this server so answers flow.
				const string name = reply.substr(6);
				send_line("{\"ok\":true,\"did\":\"talk\",\"starting\":true}");
				const bool ok = LLM_agent::begin_conversation(name);
				// After the conversation ends, notify the client.
				send_line(ok ? "{\"event\":\"conversation_ended\"}"
							  : "{\"event\":\"conversation_failed\"}");
			} else {
				send_line(reply);
			}
		}
	}

	void Agent_server_close() {
		close_client();
		if (g_listen != AGENT_INVALID_SOCKET) {
			AGENT_CLOSESOCKET(g_listen);
			g_listen = AGENT_INVALID_SOCKET;
		}
#ifdef _WIN32
		if (g_wsa) {
			WSACleanup();
			g_wsa = false;
		}
#endif
	}

	bool Agent_server_running() {
		return g_listen != AGENT_INVALID_SOCKET;
	}

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */
