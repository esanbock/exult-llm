/*
 *  agent.cc - LLM agent bridge for Exult.
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

#include "agent.h"

#include "actors.h"
#include "gamewin.h"
#include "ucmachine.h"
#include "conversation.h"
#include "tiles.h"
#include "objs.h"
#include "find_nearby.h"
#include "gamemap.h"
#include "keyactions.h"
#include "exult_constants.h"
#include "party.h"
#include "Gump_manager.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using std::string;

namespace {

	// ------------------------------------------------------------------
	//  Minimal JSON output helpers.
	// ------------------------------------------------------------------

	string json_escape(const string& s) {
		string out;
		out.reserve(s.size() + 8);
		for (const char c : s) {
			switch (c) {
			case '"':
				out += "\\\"";
				break;
			case '\\':
				out += "\\\\";
				break;
			case '\n':
				out += "\\n";
				break;
			case '\r':
				out += "\\r";
				break;
			case '\t':
				out += "\\t";
				break;
			default:
				if (static_cast<unsigned char>(c) < 0x20) {
					char buf[8];
					std::snprintf(buf, sizeof(buf), "\\u%04x", c);
					out += buf;
				} else {
					out += c;
				}
				break;
			}
		}
		return out;
	}

	string json_str(const string& key, const string& val) {
		return "\"" + key + "\":\"" + json_escape(val) + "\"";
	}

	string json_int(const string& key, long val) {
		return "\"" + key + "\":" + std::to_string(val);
	}

	string json_bool(const string& key, bool val) {
		return "\"" + key + "\":" + (val ? "true" : "false");
	}

	// ------------------------------------------------------------------
	//  Minimal JSON input helpers.  These are intentionally tiny: the
	//  request protocol is machine-generated and shallow, so we only need
	//  to pull scalar values out of a flat-ish object by key.  This is not
	//  a general JSON parser and does not validate the whole document.
	// ------------------------------------------------------------------

	// Find the raw value text following "key": in json, starting at pos.
	// Returns true and sets [vstart,vend) to the value span if found.
	bool find_value(const string& json, const string& key, size_t& vstart, size_t& vend) {
		const string needle = "\"" + key + "\"";
		size_t       k      = json.find(needle);
		while (k != string::npos) {
			size_t c = json.find(':', k + needle.size());
			if (c == string::npos) {
				return false;
			}
			size_t p = c + 1;
			while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) {
				++p;
			}
			if (p >= json.size()) {
				return false;
			}
			vstart = p;
			if (json[p] == '"') {    // string value
				size_t q = p + 1;
				while (q < json.size() && json[q] != '"') {
					if (json[q] == '\\') {
						++q;
					}
					++q;
				}
				vend = std::min(q + 1, json.size());
			} else if (json[p] == '{') {    // object value: match braces
				int    depth = 0;
				size_t q     = p;
				bool   instr = false;
				for (; q < json.size(); ++q) {
					const char ch = json[q];
					if (instr) {
						if (ch == '\\') {
							++q;
						} else if (ch == '"') {
							instr = false;
						}
					} else if (ch == '"') {
						instr = true;
					} else if (ch == '{') {
						++depth;
					} else if (ch == '}') {
						if (--depth == 0) {
							++q;
							break;
						}
					}
				}
				vend = std::min(q, json.size());
			} else {    // number/bool/null: up to , } ] or whitespace
				size_t q = p;
				while (q < json.size() && json[q] != ',' && json[q] != '}' && json[q] != ']'
					   && !std::isspace(static_cast<unsigned char>(json[q]))) {
					++q;
				}
				vend = q;
			}
			return true;
		}
		return false;
	}

	// Get a string field's decoded value (quotes stripped, basic escapes).
	bool get_string(const string& json, const string& key, string& out) {
		size_t s;
		size_t e;
		if (!find_value(json, key, s, e) || s >= json.size() || json[s] != '"') {
			return false;
		}
		out.clear();
		for (size_t i = s + 1; i + 1 <= e - 1 && i < json.size(); ++i) {
			char c = json[i];
			if (c == '"') {
				break;
			}
			if (c == '\\' && i + 1 < json.size()) {
				char n = json[++i];
				switch (n) {
				case 'n':
					out += '\n';
					break;
				case 't':
					out += '\t';
					break;
				case 'r':
					out += '\r';
					break;
				default:
					out += n;
					break;
				}
			} else {
				out += c;
			}
		}
		return true;
	}

	bool get_int(const string& json, const string& key, long& out) {
		size_t s;
		size_t e;
		if (!find_value(json, key, s, e)) {
			return false;
		}
		string tok = json.substr(s, e - s);
		if (!tok.empty() && tok.front() == '"') {
			tok = tok.substr(1, tok.size() >= 2 ? tok.size() - 2 : 0);
		}
		try {
			out = std::stol(tok);
		} catch (...) {
			return false;
		}
		return true;
	}

	// Extract the raw JSON text of a nested object field (for "action").
	bool get_object(const string& json, const string& key, string& out) {
		size_t s;
		size_t e;
		if (!find_value(json, key, s, e) || s >= json.size() || json[s] != '{') {
			return false;
		}
		out = json.substr(s, e - s);
		return true;
	}

	// ------------------------------------------------------------------
	//  ASCII map grid.
	// ------------------------------------------------------------------

	// Build a top-down character grid of the tiles around the avatar so the
	// LLM can reason about geometry and navigate around obstacles without any
	// image/OCR.  North is up (decreasing ty), east is right (increasing tx).
	//
	// Legend:
	//   @  the avatar (always at center)
	//   &  a living NPC
	//   x  a dead NPC / body
	//   *  a takeable/interactable object
	//   #  a blocked/impassable tile (wall, gate, water, furniture)
	//   .  open, walkable ground
	std::string build_grid(Actor* av, int radius) {
		Game_window* gwin = Game_window::get_instance();
		Game_map*    gmap = gwin ? gwin->get_map() : nullptr;
		if (!av || !gmap) {
			return std::string();
		}
		const Tile_coord at  = av->get_tile();
		const int        dim = 2 * radius + 1;

		// Start from terrain/blocking, then overlay objects and NPCs.
		std::vector<std::string> rows(dim, std::string(dim, '.'));

		// Blocking layer.
		for (int dy = -radius; dy <= radius; ++dy) {
			for (int dx = -radius; dx <= radius; ++dx) {
				const int tx = (at.tx + dx + c_num_tiles) % c_num_tiles;
				const int ty = (at.ty + dy + c_num_tiles) % c_num_tiles;
				const Tile_coord probe(tx, ty, at.tz);
				if (gmap->is_tile_occupied(probe)) {
					rows[dy + radius][dx + radius] = '#';
				}
			}
		}

		auto plot = [&](int tx, int ty, char c) {
			int dx = tx - at.tx;
			int dy = ty - at.ty;
			// Handle world wrap for the shortest delta.
			if (dx > c_num_tiles / 2) {
				dx -= c_num_tiles;
			} else if (dx < -c_num_tiles / 2) {
				dx += c_num_tiles;
			}
			if (dy > c_num_tiles / 2) {
				dy -= c_num_tiles;
			} else if (dy < -c_num_tiles / 2) {
				dy += c_num_tiles;
			}
			if (dx < -radius || dx > radius || dy < -radius || dy > radius) {
				return;
			}
			rows[dy + radius][dx + radius] = c;
		};

		// Object layer.
		{
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, radius, 0);
			for (Game_object* obj : objs) {
				if (!obj || obj->as_actor()) {
					continue;
				}
				const Shape_info& info = obj->get_info();
				const Tile_coord  ot   = obj->get_tile();
				if (info.is_door()) {
					// '+' = closed door (can be opened), '/' = open door.
					const bool closed = (obj->get_framenum() % 4) < 2;
					plot(ot.tx, ot.ty, closed ? '+' : '/');
					continue;
				}
				if (obj->get_name().empty()) {
					continue;
				}
				plot(ot.tx, ot.ty, '*');
			}
		}

		// NPC layer (on top).
		{
			std::vector<Actor*> npcs;
			gwin->get_nearby_npcs(npcs);
			for (Actor* npc : npcs) {
				if (!npc || npc == av) {
					continue;
				}
				const Tile_coord nt = npc->get_tile();
				plot(nt.tx, nt.ty, npc->is_dead() ? 'x' : '&');
			}
		}

		// Avatar at center.
		rows[radius][radius] = '@';

		// Join rows with \n.
		std::string out;
		out.reserve(dim * (dim + 1));
		for (int i = 0; i < dim; ++i) {
			if (i) {
				out += '\n';
			}
			out += rows[i];
		}
		return out;
	}

	// ------------------------------------------------------------------
	//  Synthetic input injection (mirrors android/TouchUI_Android.cc).
	// ------------------------------------------------------------------

	void push_key(SDL_Keycode key) {
		SDL_Event event = {};
		event.key.key   = key;
		event.type      = SDL_EVENT_KEY_DOWN;
		SDL_PushEvent(&event);
		event.type = SDL_EVENT_KEY_UP;
		SDL_PushEvent(&event);
	}

	// Map a friendly key name to an SDL keycode.  Returns 0 if unknown.
	SDL_Keycode keycode_for(const string& name) {
		if (name.size() == 1) {
			const unsigned char c = static_cast<unsigned char>(name[0]);
			if (std::isalnum(c)) {
				return static_cast<SDL_Keycode>(std::tolower(c));
			}
		}
		if (name == "space") {
			return SDLK_SPACE;
		}
		if (name == "escape" || name == "esc") {
			return SDLK_ESCAPE;
		}
		if (name == "return" || name == "enter") {
			return SDLK_RETURN;
		}
		return 0;
	}

}    // namespace

namespace LLM_agent {

	string observe() {
		std::ostringstream os;
		os << '{';

		Game_window* gwin = Game_window::get_instance();
		if (!gwin) {
			os << json_bool("world_loaded", false) << '}';
			return os.str();
		}

		Actor* av = gwin->get_main_actor();
		os << json_bool("world_loaded", av != nullptr);

		if (av) {
			const Tile_coord t = av->get_tile();
			os << ',' << "\"player\":{";
			os << json_str("name", av->get_name());
			os << ',' << json_int("tx", t.tx);
			os << ',' << json_int("ty", t.ty);
			os << ',' << json_int("tz", t.tz);
			os << ',' << json_int("hp", av->get_property(Actor::health));
			os << ',' << json_int("str", av->get_property(Actor::strength));
			os << ',' << json_int("dex", av->get_property(Actor::dexterity));
			os << ',' << json_int("int", av->get_property(Actor::intelligence));
			os << ',' << json_int("mana", av->get_property(Actor::mana));
			os << ',' << json_int("food", av->get_property(Actor::food_level));
			os << ',' << json_bool("dead", av->is_dead());
			os << '}';
		}

		os << ',' << json_bool("in_combat", gwin->in_combat());
		os << ',' << json_bool("moving", gwin->is_moving());
		os << ',' << json_bool("in_dungeon", gwin->is_in_dungeon() != 0);

		// Nearby NPCs.
		os << ',' << "\"nearby\":[";
		{
			std::vector<Actor*> list;
			gwin->get_nearby_npcs(list);
			bool                first = true;
			const Tile_coord    at    = av ? av->get_tile() : Tile_coord(0, 0, 0);
			int                 count = 0;
			for (Actor* npc : list) {
				if (!npc || npc == av) {
					continue;
				}
				if (count++ >= 24) {    // cap payload size
					break;
				}
				const Tile_coord nt = npc->get_tile();
				if (!first) {
					os << ',';
				}
				first = false;
				os << '{';
				os << json_str("name", npc->get_name());
				os << ',' << json_int("tx", nt.tx);
				os << ',' << json_int("ty", nt.ty);
				os << ',' << json_int("dx", nt.tx - at.tx);
				os << ',' << json_int("dy", nt.ty - at.ty);
				os << ',' << json_bool("in_party", npc->get_party_id() >= 0);
				os << ',' << json_bool("dead", npc->is_dead());
				os << '}';
			}
		}
		os << ']';

		// Nearby interactable objects (non-NPC items on screen), by name.
		os << ',' << "\"objects\":[";
		if (av) {
			Game_object_vector objs;
			const Tile_coord   at = av->get_tile();
			Game_object::find_nearby(objs, at, -1, 12, 0);
			bool first = true;
			int  count = 0;
			for (Game_object* obj : objs) {
				if (!obj) {
					continue;
				}
				if (obj->as_actor()) {
					continue;    // NPCs are already in "nearby"
				}
				const std::string nm = obj->get_name();
				if (nm.empty()) {
					continue;
				}
				if (count++ >= 24) {
					break;
				}
				const Tile_coord ot = obj->get_tile();
				if (!first) {
					os << ',';
				}
				first = false;
				os << '{';
				os << json_str("name", nm);
				os << ',' << json_int("dx", ot.tx - at.tx);
				os << ',' << json_int("dy", ot.ty - at.ty);
				os << '}';
			}
		}
		os << ']';

		// Top-down ASCII map grid centered on the avatar.
		{
			const int   radius = 12;
			std::string grid   = build_grid(av, radius);
			os << ',' << json_int("grid_radius", radius);
			os << ',' << json_str("grid_legend",
					"@=you &=npc x=body *=object +=closed_door /=open_door #=blocked .=open; north=up east=right");
			os << ',' << json_str("grid", grid);
		}

		// Nearby doors (with open/closed state) so the agent can plan routes
		// and know when to "open" a door to pass through.
		os << ',' << "\"doors\":[";
		if (av) {
			Game_object_vector objs;
			const Tile_coord   at = av->get_tile();
			Game_object::find_nearby(objs, at, -1, 12, 0);
			bool first = true;
			int  count = 0;
			for (Game_object* obj : objs) {
				if (!obj || !obj->get_info().is_door()) {
					continue;
				}
				if (count++ >= 12) {
					break;
				}
				const Tile_coord ot     = obj->get_tile();
				const bool       closed = (obj->get_framenum() % 4) < 2;
				if (!first) {
					os << ',';
				}
				first = false;
				os << '{';
				os << json_str("name", obj->get_name());
				os << ',' << json_int("dx", ot.tx - at.tx);
				os << ',' << json_int("dy", ot.ty - at.ty);
				os << ',' << json_bool("closed", closed);
				os << '}';
			}
		}
		os << ']';

		// Active conversation.
		Usecode_machine* uc   = gwin->get_usecode();
		Conversation*    conv = uc ? uc->get_conversation() : nullptr;
		const bool       in_progress = conv && conv->get_num_faces_on_screen() > 0;
		const bool       convo_active
				= conv && conv->are_choices_active() && conv->get_num_answers() > 0;
		os << ',' << json_bool("conversation_in_progress", in_progress);
		os << ',' << json_bool("conversation_active", convo_active);
		os << ',' << "\"answers\":[";
		if (convo_active) {
			const int n = conv->get_num_answers();
			for (int i = 0; i < n; ++i) {
				if (i) {
					os << ',';
				}
				const char* a = conv->get_answer(i);
				os << '"' << json_escape(a ? a : "") << '"';
			}
		}
		os << ']';
		// The most recent line the NPC spoke, if any.
		os << ',' << json_str("npc_text", conv ? conv->get_last_npc_text() : std::string());

		os << '}';
		return os.str();
	}

	string act(const string& action_json) {
		string type;
		if (!get_string(action_json, "type", type)) {
			return "{\"ok\":false,\"error\":\"missing action type\"}";
		}

		Game_window* gwin = Game_window::get_instance();
		if (!gwin) {
			return "{\"ok\":false,\"error\":\"no game window\"}";
		}

		if (type == "wait") {
			return "{\"ok\":true,\"did\":\"wait\"}";
		}

		if (type == "stop") {
			gwin->stop_actor();
			return "{\"ok\":true,\"did\":\"stop\"}";
		}

		if (type == "key") {
			string keyname;
			if (!get_string(action_json, "key", keyname)) {
				return "{\"ok\":false,\"error\":\"missing key\"}";
			}
			SDL_Keycode kc = keycode_for(keyname);
			if (kc == 0) {
				return "{\"ok\":false,\"error\":\"unknown key\"}";
			}
			push_key(kc);
			return "{\"ok\":true,\"did\":\"key\",\"key\":\"" + json_escape(keyname) + "\"}";
		}

		if (type == "answer") {
			Usecode_machine* uc   = gwin->get_usecode();
			Conversation*    conv = uc ? uc->get_conversation() : nullptr;
			if (!conv || !conv->are_choices_active() || conv->get_num_answers() <= 0) {
				return "{\"ok\":false,\"error\":\"no active conversation\"}";
			}
			int idx = -1;
			long li;
			string text;
			if (get_int(action_json, "index", li)) {
				idx = static_cast<int>(li);
			} else if (get_string(action_json, "text", text)) {
				idx = conv->locate_answer(text.c_str());
			}
			if (idx < 0 || idx >= conv->get_num_answers()) {
				return "{\"ok\":false,\"error\":\"answer out of range\"}";
			}
			// The conversation input loop accepts keys '1'..'9' then 'a'..
			constexpr static const char keys[] = "123456789abcdefghijklmnopqrstuvwxyz";
			if (idx < static_cast<int>(sizeof(keys) - 1)) {
				push_key(static_cast<SDL_Keycode>(keys[idx]));
				return "{\"ok\":true,\"did\":\"answer\"," + json_int("index", idx) + "}";
			}
			return "{\"ok\":false,\"error\":\"answer index too large\"}";
		}

		if (type == "combat") {
			ActionCombat(nullptr);
			return "{\"ok\":true,\"did\":\"combat\"}";
		}

		if (type == "combat_pause") {
			ActionCombatPause(nullptr);
			return "{\"ok\":true,\"did\":\"combat_pause\"}";
		}

		if (type == "inventory") {
			long member = -1;
			get_int(action_json, "member", member);
			int p[1] = {static_cast<int>(member)};
			ActionInventory(p);
			return "{\"ok\":true,\"did\":\"inventory\"}";
		}

		if (type == "stats") {
			int p[1] = {-1};
			ActionStats(p);
			return "{\"ok\":true,\"did\":\"stats\"}";
		}

		if (type == "feed") {
			// Restore food level directly (no usecode / no blocking).  Feeds
			// the whole party by default; the avatar only if "avatar_only".
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			long level = 30;    // U7 food level maxes out around 30.
			get_int(action_json, "level", level);
			if (level < 0) {
				level = 0;
			}
			if (level > 30) {
				level = 30;
			}
			bool avatar_only = false;
			{
				string tmp;
				if (get_string(action_json, "avatar_only", tmp)) {
					avatar_only = (tmp == "true" || tmp == "1");
				}
			}
			int fed = 0;
			av->set_property(static_cast<int>(Actor::food_level), static_cast<int>(level));
			fed = 1;
			if (!avatar_only) {
				Party_manager* pm  = gwin->get_party_man();
				const int      cnt = pm ? pm->get_count() : 0;
				for (int i = 0; i < cnt; ++i) {
					Actor* member = gwin->get_npc(pm->get_member(i));
					if (member && !member->is_dead()) {
						member->set_property(static_cast<int>(Actor::food_level), static_cast<int>(level));
						++fed;
					}
				}
			}
			return "{\"ok\":true,\"did\":\"feed\"," + json_int("level", level) + ","
				   + json_int("count", fed) + "}";
		}

		if (type == "heal") {
			// Only the non-blocking path: try to apply a bandage.  (The full
			// ActionUseHealingItems also falls back to a click-to-target
			// potion, which would block the agent, so we avoid it here.)
			const bool ok = gwin->activate_item(827);    // bandage
			return ok ? "{\"ok\":true,\"did\":\"heal\",\"item\":\"bandage\"}"
					  : "{\"ok\":false,\"error\":\"no bandage in party\"}";
		}

		if (type == "open") {
			// Open (toggle) the nearest door within a few tiles.  Optional
			// {"dir":...} biases toward a door in that compass direction.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			const Tile_coord   at = av->get_tile();
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, 4, 0);
			Game_object* best   = nullptr;
			int          best_d = 1 << 30;
			for (Game_object* obj : objs) {
				if (!obj || !obj->get_info().is_door()) {
					continue;
				}
				const Tile_coord ot = obj->get_tile();
				const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
				if (d < best_d) {
					best_d = d;
					best   = obj;
				}
			}
			if (!best) {
				return "{\"ok\":false,\"error\":\"no door nearby\"}";
			}
			best->activate();    // toggles open/closed
			return "{\"ok\":true,\"did\":\"open\"}";
		}

		if (type == "move") {
			string dir;
			get_string(action_json, "dir", dir);
			long speed = 200;
			get_int(action_json, "speed", speed);
			const int w = gwin->get_width();
			const int h = gwin->get_height();
			// Target a screen point offset from center in the requested
			// compass direction (screen y grows downward; north = up).
			const int cx  = w / 2;
			const int cy  = h / 2;
			const int off = 50;
			int       tx  = cx;
			int       ty  = cy;
			if (dir == "n") {
				ty = cy - off;
			} else if (dir == "s") {
				ty = cy + off;
			} else if (dir == "e") {
				tx = cx + off;
			} else if (dir == "w") {
				tx = cx - off;
			} else if (dir == "ne") {
				tx = cx + off;
				ty = cy - off;
			} else if (dir == "nw") {
				tx = cx - off;
				ty = cy - off;
			} else if (dir == "se") {
				tx = cx + off;
				ty = cy + off;
			} else if (dir == "sw") {
				tx = cx - off;
				ty = cy + off;
			} else {
				return "{\"ok\":false,\"error\":\"bad direction\"}";
			}
			gwin->start_actor(tx, ty, static_cast<int>(speed));
			return "{\"ok\":true,\"did\":\"move\",\"dir\":\"" + json_escape(dir) + "\"}";
		}

		return "{\"ok\":false,\"error\":\"unknown action type\"}";
	}

	string handle_request(const string& request_json) {
		string cmd;
		if (!get_string(request_json, "cmd", cmd)) {
			return "{\"ok\":false,\"error\":\"missing cmd\"}";
		}
		if (cmd == "observe") {
			return observe();
		}
		if (cmd == "act") {
			string action;
			if (!get_object(request_json, "action", action)) {
				return "{\"ok\":false,\"error\":\"missing action object\"}";
			}
			return act(action);
		}
		if (cmd == "talk") {
			// Return a sentinel; the server acks then calls begin_conversation
			// so the client is free to send "answer" actions during the
			// (blocking) conversation loop.
			string name;
			get_string(request_json, "name", name);
			return "@TALK@" + name;
		}
		if (cmd == "ping") {
			return "{\"ok\":true,\"pong\":true}";
		}
		return "{\"ok\":false,\"error\":\"unknown cmd\"}";
	}

	bool begin_conversation(const string& name) {
		Game_window* gwin = Game_window::get_instance();
		if (!gwin) {
			return false;
		}
		Actor* av = gwin->get_main_actor();
		if (!av) {
			return false;
		}
		// Find the target NPC: matching name if given, else the nearest.
		std::vector<Actor*> npcs;
		gwin->get_nearby_npcs(npcs);
		const Tile_coord at = av->get_tile();
		Actor*           best = nullptr;
		int              best_d = 1 << 30;
		for (Actor* npc : npcs) {
			if (!npc || npc == av || npc->is_dead()) {
				continue;
			}
			if (!name.empty()) {
				// Case-insensitive substring match on the NPC's name.
				string nm = npc->get_name();
				string a  = nm;
				string b  = name;
				std::transform(a.begin(), a.end(), a.begin(), ::tolower);
				std::transform(b.begin(), b.end(), b.begin(), ::tolower);
				if (a.find(b) == string::npos) {
					continue;
				}
			}
			const Tile_coord nt = npc->get_tile();
			const int        d  = std::abs(nt.tx - at.tx) + std::abs(nt.ty - at.ty);
			if (d < best_d) {
				best_d = d;
				best   = npc;
			}
		}
		if (!best) {
			return false;
		}
		// A double-click on a party member shows their inventory instead of
		// starting a conversation when gumps are already open (or in combat).
		// Close any open gumps first so the activation converses instead.
		Gump_manager* gm = gwin->get_gump_man();
		if (gm && gm->showing_gumps(true)) {
			gm->close_all_gumps();
		}
		// Double-click activation (event 1) starts the NPC's conversation.
		best->activate(1);
		return true;
	}

}    // namespace LLM_agent

#endif /* USE_LLM_AGENT */
