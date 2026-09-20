/*
 *  agentserver.h - Local TCP JSON-line server for the LLM agent bridge.
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

#ifndef INCL_LLM_AGENTSERVER_H
#define INCL_LLM_AGENTSERVER_H 1

#ifdef USE_LLM_AGENT

/*
 *  A tiny, dependency-free, non-blocking TCP server bound to 127.0.0.1.
 *  The protocol is newline-delimited JSON: each line the client sends is a
 *  request object (see LLM_agent::handle_request), and the server replies
 *  with exactly one JSON line per request.
 *
 *  It is driven cooperatively from the main event loop: call
 *  Agent_server_init() once after the window is up, Agent_server_poll()
 *  once per frame, and Agent_server_close() at shutdown.
 */
namespace LLM_agent {

	// Start listening on 127.0.0.1:<port>.  Returns true on success.
	bool Agent_server_init(int port = 0);

	// Service pending connections/requests without blocking.  Safe to call
	// even if the server failed to start (no-op in that case).
	void Agent_server_poll();

	// Shut the server down and release resources.
	void Agent_server_close();

	// True if the server socket is currently listening.
	bool Agent_server_running();

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */

#endif /* INCL_LLM_AGENTSERVER_H */
