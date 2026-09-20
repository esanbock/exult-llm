/*
 *  agent.h - LLM agent bridge for Exult.
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

#ifndef INCL_LLM_AGENT_H
#define INCL_LLM_AGENT_H 1

#ifdef USE_LLM_AGENT

#include <string>

/*
 *  The LLM agent bridge.  This is a thin façade the network server calls
 *  into.  It reads the live game state (via Game_window) and serializes it
 *  to a compact JSON object, and it executes discrete actions requested by
 *  an external driver by pushing synthetic SDL input events into the queue
 *  (the same mechanism the touch UI uses).
 *
 *  All methods are safe to call from the main thread only; the network
 *  server marshals requests onto the main loop.
 */
namespace LLM_agent {

	// Serialize the current game state to a JSON object (single line, no
	// trailing newline).  Never throws; returns a valid JSON object even
	// when the world is not yet loaded (fields become null/empty).
	std::string observe();

	// Execute a single action described by a JSON request object (the value
	// of the "action" field sent by the driver).  Returns a JSON result
	// object describing success/failure.  Recognized actions:
	//    {"type":"move","dir":"n|s|e|w|ne|nw|se|sw","speed":<ms>}
	//    {"type":"stop"}
	//    {"type":"key","key":"<name>"}     e.g. "space","escape","1".."9","a"
	//    {"type":"answer","index":<n>}     select conversation answer n (0-based)
	//    {"type":"answer","text":"bye"}    select conversation answer by text
	//    {"type":"wait"}                   no-op (let the world tick)
	std::string act(const std::string& action_json);

	// Handle a full request line of the form {"cmd":"observe"} or
	// {"cmd":"act","action":{...}}.  Returns the JSON response line.
	std::string handle_request(const std::string& request_json);

	// Start a conversation with a nearby living NPC by name (or the nearest
	// NPC if name is empty).  Returns true if a conversation was started.
	// NOTE: this runs the NPC's usecode, which may block in the engine's
	// answer-selection loop until the client sends "answer" actions; that
	// loop polls the agent server, so answers still flow.  The server calls
	// this AFTER acking the request so the client isn't blocked.
	bool begin_conversation(const std::string& name);
}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */

#endif /* INCL_LLM_AGENT_H */
