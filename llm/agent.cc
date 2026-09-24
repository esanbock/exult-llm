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
#include "schedule.h"
#include "gameclk.h"
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
#include "Sign_gump.h"
#include "Slider_gump.h"
#include "contain.h"
#include "objiter.h"
#include "ready.h"
#include "effects.h"
#include "utils.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
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
	std::string build_grid(Actor* av, int rx, int ry) {
		Game_window* gwin = Game_window::get_instance();
		Game_map*    gmap = gwin ? gwin->get_map() : nullptr;
		if (!av || !gmap) {
			return std::string();
		}
		const Tile_coord at   = av->get_tile();
		const int        dimx = 2 * rx + 1;
		const int        dimy = 2 * ry + 1;

		// Start from terrain/blocking, then overlay objects and NPCs.
		std::vector<std::string> rows(dimy, std::string(dimx, '#'));

		// Walkability via 3D REACHABILITY BFS from the avatar. The world is 3D:
		// stairs raise you to a wall-top / upper floor and only climb from the
		// correct approach tile. A single-Z occupancy test (or a one-step check)
		// mis-marks wall-tops as solid and can't tell that a stairs tile is
		// reachable only from its bottom step. So we FLOOD-FILL using the
		// avatar's own step primitive (Actor::is_blocked), which returns the
		// resulting standing height (new_lift) for each step - naturally
		// climbing stairs, walking elevated walkways, and refusing side
		// approaches. A tile is '.' (walkable) iff the BFS can stand on it.
		const int move_flags = av->get_type_flags();
		{
			// Per-cell best-known standing tz reached by the BFS (-128 = unseen).
			std::vector<int> reach(dimx * dimy, -128);
			auto idx = [&](int gx, int gy) { return gy * dimx + gx; };
			std::deque<std::pair<int, int>> bfs;  // grid (gx,gy)
			const int cxg = rx, cyg = ry;         // avatar at grid center
			reach[idx(cxg, cyg)] = at.tz;
			bfs.emplace_back(cxg, cyg);
			rows[cyg][cxg] = '.';
			static const int ndx[8] = {0, 0, 1, -1, 1, 1, -1, -1};
			static const int ndy[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
			while (!bfs.empty()) {
				auto [gx, gy] = bfs.front();
				bfs.pop_front();
				const int fz = reach[idx(gx, gy)];
				const Tile_coord from(
						(at.tx + (gx - cxg) + c_num_tiles) % c_num_tiles,
						(at.ty + (gy - cyg) + c_num_tiles) % c_num_tiles, fz);
				for (int d = 0; d < 8; ++d) {
					const int nx = gx + ndx[d];
					const int ny = gy + ndy[d];
					if (nx < 0 || nx >= dimx || ny < 0 || ny >= dimy) {
						continue;
					}
					if (reach[idx(nx, ny)] != -128) {
						continue;    // already reached
					}
					Tile_coord to(
							(at.tx + (nx - cxg) + c_num_tiles) % c_num_tiles,
							(at.ty + (ny - cyg) + c_num_tiles) % c_num_tiles, fz);
					Tile_coord fromc = from;
					const bool blocked = av->is_blocked(to, &fromc, move_flags);
					if (!blocked) {
						reach[idx(nx, ny)] = to.tz;   // resulting standing height
						rows[ny][nx] = '.';
						bfs.emplace_back(nx, ny);
					}
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
			if (dx < -rx || dx > rx || dy < -ry || dy > ry) {
				return;
			}
			rows[dy + ry][dx + rx] = c;
		};

		// Object layer.
		{
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, (rx > ry ? rx : ry), 128);
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
				// Glyphs earn their place ONLY if they add navigational value
				// the "objects" list can't: walkability, routes, and loot/exit
				// locations. Item IDENTITY lives in the objects list, so we do
				// NOT spend glyphs distinguishing tree/wall/furniture/sign - they
				// are all just BLOCKED tiles for movement and already show as '#'
				// from the blocking layer. We only override '#' when the glyph
				// tells the agent something actionable about that tile.
				const std::string nm  = obj->get_name();
				std::string        low = nm;
				std::transform(low.begin(), low.end(), low.begin(), ::tolower);
				const bool is_exit =
						(low.find("portcullis") != std::string::npos
						 || low.find("gateway") != std::string::npos
						 || low.find("stair") != std::string::npos
						 || low.find("ladder") != std::string::npos
						 || low.find("trapdoor") != std::string::npos
						 || (low.find("gate") != std::string::npos
							 && low.find("fence") == std::string::npos));
				char g = 0;    // 0 => leave the underlying terrain/'#' as-is
				if (is_exit) {
					g = 'E';    // an EXIT/route: town gate, stairs, ladder
				} else if (info.is_body_shape()) {
					// Lootable (container) body vs a corpse with nothing to take.
					g = obj->as_container() ? 'b' : 'x';
				} else if (info.is_water()) {
					g = '~';    // water (blocks walking)
				} else if (info.get_shape_class() == Shape_info::container) {
					g = 'n';    // container you can search for loot
				} else if (low.find("fence") != std::string::npos
						   || low.find("rail") != std::string::npos) {
					g = '=';    // linear barrier (look for a gap/gate)
				} else if (info.get_shape_class() == Shape_info::building
						   || low.find("tree") != std::string::npos
						   || info.is_solid()) {
					// Blocking structure/scenery reads as a wall so the map is
					// about walkability; identity (if notable) is in objects[].
					g = '#';
				} else {
					g = '*';    // a loose named item on the ground (pickup-able)
				}
				if (g) {
					plot(ot.tx, ot.ty, g);
				}
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
				char glyph;
				if (npc->is_dead()) {
					// Lootable corpse (container) vs one with nothing to take.
					glyph = npc->as_container() ? 'b' : 'x';
				} else if (npc->get_party_id() >= 0) {
					glyph = 'C';                       // party companion
				} else {
					glyph = '&';                       // other NPC
				}
				plot(nt.tx, nt.ty, glyph);
			}
		}

		// Avatar at center.
		rows[ry][rx] = '@';

		// Join rows with \n.
		std::string out;
		out.reserve(dimy * (dimx + 1));
		for (int i = 0; i < dimy; ++i) {
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

	// Most recent sign/plaque text, populated by the usecode display_runes
	// intrinsic (via LLM_agent_set_last_sign_text) when a sign is shown, so the
	// "read" action can return it even though the sign gump is modal and
	// auto-dismissed.
	std::string g_llm_last_sign_text;

	string screenshot();    // fwd decl (defined after handle_request)

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
			// Max HP == strength in U7 (health is capped at strength). Emit it
			// so the agent knows current-vs-max and doesn't think full HP is low.
			os << ',' << json_int("max_hp", av->get_property(Actor::strength));
			os << ',' << json_int("str", av->get_property(Actor::strength));
			os << ',' << json_int("dex", av->get_property(Actor::dexterity));
			os << ',' << json_int("int", av->get_property(Actor::intelligence));
			os << ',' << json_int("mana", av->get_property(Actor::mana));
			os << ',' << json_int("food", av->get_property(Actor::food_level));
			os << ',' << json_bool("dead", av->is_dead());
			os << '}';
		}

		// Time of day - so the agent knows when NPCs sleep/are available and can
		// choose to wait for morning.
		if (Game_clock* clk = gwin->get_clock()) {
			const int hr = clk->get_hour();
			os << ',' << json_int("hour", hr);
			os << ',' << json_int("minute", clk->get_minute());
			os << ',' << json_int("day", clk->get_day());
			const char* part = (hr < 6) ? "night" : (hr < 12) ? "morning"
					: (hr < 18) ? "afternoon" : (hr < 21) ? "evening" : "night";
			os << ',' << json_str("time_of_day", part);
			// Most townsfolk sleep roughly 21:00-06:00.
			os << ',' << json_bool("is_night", hr >= 21 || hr < 6);
		}

		os << ',' << json_bool("in_combat", gwin->in_combat());
		os << ',' << json_bool("moving", gwin->is_moving());
		os << ',' << json_bool("in_dungeon", gwin->is_in_dungeon() != 0);
		// A container/body gump (or menu) is open, which blocks movement until
		// you take what you want and "close" it.
		{
			Gump_manager* gm = gwin->get_gump_man();
			const bool open = gm && gm->showing_gumps(true);
			os << ',' << json_bool("gump_open", open);
			// Report the contents of the container whose gump is ACTUALLY open,
			// so the driver loots the right thing (not some other nearby bag).
			// Scan nearby objects; the one with an open gump is the open one.
			os << ',' << "\"gump_contents\":[";
			if (open && av) {
				const Tile_coord at = av->get_tile();
				Game_object_vector cobjs;
				Game_object::find_nearby(cobjs, at, -1, 6, 128);
				Container_game_object* opencont = nullptr;
				for (Game_object* obj : cobjs) {
					if (!obj) {
						continue;
					}
					Container_game_object* cc = obj->as_container();
					if (cc && gm->find_gump(obj)) {
						opencont = cc;
						break;
					}
				}
				if (opencont) {
					bool cfirst = true;
					Object_iterator it(opencont->get_objects());
					Game_object* inner;
					int ccount = 0;
					while ((inner = it.get_next()) != nullptr && ccount < 30) {
						const std::string inm = inner->get_name();
						if (inm.empty()) {
							continue;
						}
						if (!cfirst) {
							os << ',';
						}
						cfirst = false;
						os << '"' << json_escape(inm) << '"';
						++ccount;
					}
				}
			}
			os << ']';
		}

		// Nearby NPCs.
		os << ',' << "\"nearby\":[";
		{
			const Tile_coord at = av ? av->get_tile() : Tile_coord(0, 0, 0);
			// Use a STABLE radius-based actor scan rather than the volatile
			// proximity manager (get_nearby_npcs flickered turn-to-turn and
			// reported NPCs from far off inconsistently). This gives the agent a
			// consistent, screen-like view of who is nearby - what a human sees.
			Actor_vector actors;
			if (av) {
				Game_object::find_nearby_actors(actors, at, c_any_shapenum, 24);
			}
			std::vector<std::pair<int, Actor*>> sorted;
			for (Actor* npc : actors) {
				if (!npc || npc == av) {
					continue;
				}
				const Tile_coord nt = npc->get_tile();
				sorted.emplace_back(std::abs(nt.tx - at.tx) + std::abs(nt.ty - at.ty), npc);
			}
			std::sort(sorted.begin(), sorted.end(),
					  [](const auto& a, const auto& b) { return a.first < b.first; });
			bool first = true;
			int  count = 0;
			for (auto& [d, npc] : sorted) {
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
				// Relative elevation: dz>0 = higher than you, dz<0 = lower.
				// same_level=false means this NPC is on a different floor/level
				// and is NOT directly reachable without changing elevation.
				os << ',' << json_int("dz", nt.tz - at.tz);
				os << ',' << json_bool("same_level", nt.tz == at.tz);
				os << ',' << json_bool("in_party", npc->get_party_id() >= 0);
				os << ',' << json_bool("dead", npc->is_dead());
				// Status the agent should know before trying to interact - most
				// importantly SLEEPING (can't be talked to). Report the most
				// relevant single status word.
				{
					const char* st = nullptr;
					if (npc->get_flag(Obj_flags::asleep)
							|| npc->get_schedule_type() == Schedule::sleep) {
						st = "sleeping";
					} else if (npc->get_flag(Obj_flags::paralyzed)) {
						st = "paralyzed";
					} else if (npc->get_flag(Obj_flags::poisoned)) {
						st = "poisoned";
					} else if (npc->get_flag(Obj_flags::charmed)) {
						st = "charmed";
					} else if (npc->get_flag(Obj_flags::cursed)) {
						st = "cursed";
					} else if (npc->get_schedule_type() == Schedule::combat) {
						st = "hostile";
					}
					if (st) {
						os << ',' << json_str("condition", st);
					}
				}
				os << '}';
			}
		}
		os << ']';

		// Nearby interactable objects (non-NPC items on screen), by name,
		// closest first so a busy scene (e.g. a murder scene) leads with its
		// most relevant items rather than random furniture.
		os << ',' << "\"objects\":[";
		if (av) {
			Game_object_vector objs;
			const Tile_coord   at = av->get_tile();
			Game_object::find_nearby(objs, at, -1, 12, 128);
			// Collect (rank, distance, obj). We rank likely-interactable items
			// (takeable loot, containers, bodies) ahead of bulk scenery
			// (walls, roofs, fences, trees) so a scene crowded with structure
			// doesn't push the actual items (e.g. jewelry on a table) past the
			// output cap and hide them from the agent.
			std::vector<std::tuple<int, int, Game_object*>> items;
			for (Game_object* obj : objs) {
				if (!obj || obj->as_actor()) {
					continue;
				}
				if (obj->get_name().empty()) {
					continue;
				}
				const Shape_info& info = obj->get_info();
				const Tile_coord ot = obj->get_tile();
				const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
				const auto sclass = info.get_shape_class();
				std::string low = obj->get_name();
				std::transform(low.begin(), low.end(), low.begin(), ::tolower);
				auto has = [&low](const char* w) {
					return low.find(w) != std::string::npos;
				};
				// Bulk scenery: repeated structure/ground clutter. Kept but
				// deduped by name (see emission) so it never floods the list.
				const bool bulk = has("roof") || has("wall") || has("fence")
						  || has("tree") || has("floor") || has("blood")
						  || has("garbage") || has("rubble") || has("grass")
						  || has("dirt") || has("water");
				const bool scenery = info.is_solid()
						  || sclass == Shape_info::building
						  || sclass == Shape_info::unusable;
				// Notable: things worth going to / picking up / investigating.
				const bool notable
						  = info.is_body_shape() || obj->as_container()
						  || has("body") || has("victim") || has("corpse")
						  || has("chest") || has("key") || has("scroll")
						  || has("book") || has("note") || has("letter")
						  || has("gold") || has("gem") || has("ring")
						  || has("statue") || has("gargoyle") || has("jewel")
						  || has("potion") || has("wand") || has("sword")
						  || has("shield") || has("armor") || has("lever")
						  || has("switch") || has("altar") || has("shrine")
						  || has("pitchfork") || has("tongs");
				// Rank 0 = notable, 1 = normal item, 2 = bulk scenery.
				int rank = notable ? 0 : (bulk || scenery) ? 2 : 1;
				items.emplace_back(rank, d, obj);
			}
			// Sort by rank first (interesting before scenery), then distance.
			std::sort(items.begin(), items.end(),
					  [](const auto& a, const auto& b) {
						  if (std::get<0>(a) != std::get<0>(b)) {
							  return std::get<0>(a) < std::get<0>(b);
						  }
						  return std::get<1>(a) < std::get<1>(b);
					  });
			bool first = true;
			int  count = 0;
			// Count bulk-scenery occurrences by name so we can collapse repeats
			// (e.g. 9x "blood", many "wood roof") into ONE entry with a count
			// instead of flooding the list and hiding real items.
			std::map<std::string, int> scenery_seen;
			for (auto& [rank, d, obj] : items) {
				if (count >= 24) {
					break;
				}
				const std::string nm = obj->get_name();
				if (rank == 2) {
					int& n = scenery_seen[nm];
					++n;
					if (n > 1) {
						continue;    // already emitted this scenery name once
					}
				}
				++count;
				const Tile_coord  ot   = obj->get_tile();
				const Shape_info& info = obj->get_info();
				const bool is_body = info.is_body_shape();
				// A body is only LOOTABLE if it is a container (e.g. the slain
				// gargoyle you can search). A body-shape that is NOT a container
				// (e.g. a ritually-murdered corpse) has nothing to take -
				// searching it is pointless. Distinguish them so the agent
				// doesn't loop searching a non-lootable corpse.
				const bool is_lootable_body = is_body && obj->as_container();
				if (!first) {
					os << ',';
				}
				first = false;
				os << '{';
				os << json_str("name", nm);
				os << ',' << json_int("dx", ot.tx - at.tx);
				os << ',' << json_int("dy", ot.ty - at.ty);
				os << ',' << json_int("dz", ot.tz - at.tz);
				if (ot.tz != at.tz) {
					os << ',' << json_bool("same_level", false);
				}
				if (is_lootable_body) {
					os << ',' << json_bool("body", true);    // searchable/lootable
				} else if (is_body) {
					// A corpse with nothing to loot - searching does nothing.
					os << ',' << json_bool("corpse", true);
				}
				// Town EXIT: a portcullis / town gate / gateway is how you leave
				// the town to the outside world. Flag it so the agent knows this
				// is the way out (a human recognises the big gate on sight).
				{
					std::string _lo = nm;
					std::transform(_lo.begin(), _lo.end(), _lo.begin(), ::tolower);
					if (_lo.find("portcullis") != std::string::npos
							|| _lo.find("gateway") != std::string::npos
							|| (_lo.find("gate") != std::string::npos
								&& _lo.find("gateway") == std::string::npos
								&& _lo.find("fence") == std::string::npos)) {
						os << ',' << json_bool("town_exit", true);
					}
				}
				// Ownership: an item flagged okay_to_take is free to take;
				// otherwise it may be someone's property (taking it is theft).
				// Report owned=true for not-okay-to-take, non-body items so the
				// agent can avoid stealing. Bodies/containers you loot are fine.
				if (!is_body && !obj->get_flag(Obj_flags::okay_to_take)
						&& !obj->as_container()) {
					os << ',' << json_bool("owned", true);
				}
				// If this is a container (body, bag, chest...), list what is
				// inside (recursively, so loot inside a bag inside a body shows).
				Container_game_object* cont = obj->as_container();
				if (cont) {
					os << ',' << "\"contents\":[";
					bool cfirst = true;
					int  ccount = 0;
					// Simple recursive walk (bounded) of the container tree.
					std::vector<Container_game_object*> stack{cont};
					while (!stack.empty() && ccount < 20) {
						Container_game_object* c = stack.back();
						stack.pop_back();
						Object_iterator it(c->get_objects());
						Game_object* inner;
						while ((inner = it.get_next()) != nullptr && ccount < 20) {
							const std::string inm = inner->get_name();
							if (!inm.empty()) {
								if (!cfirst) {
									os << ',';
								}
								cfirst = false;
								os << '"' << json_escape(inm) << '"';
								++ccount;
							}
							if (Container_game_object* ic = inner->as_container()) {
								stack.push_back(ic);
							}
						}
					}
					os << ']';
				}
				os << '}';
			}
		}
		os << ']';

		// Top-down ASCII map grid centered on the avatar. Match the HUMAN-
		// VISIBLE tile window so we show no more than a player sees: Exult
		// renders (get_width()/c_tilesize) x (get_height()/c_tilesize) tiles
		// (see gamerend.cc). The avatar is centered, so the half-extents are
		// half of each. Cap to keep the prompt bounded on huge windows.
		{
			int vis_w = 25, vis_h = 25;
			if (gwin && gwin->get_win()) {
				vis_w = gwin->get_width() / c_tilesize;
				vis_h = gwin->get_height() / c_tilesize;
			}
			// Half-extent so the full span (2*r+1) fits within the visible
			// tiles; cap the radii so the map stays a reasonable prompt size.
			int rx = (vis_w - 1) / 2;
			int ry = (vis_h - 1) / 2;
			if (rx > 20) rx = 20;
			if (ry > 14) ry = 14;
			if (rx < 4)  rx = 4;
			if (ry < 4)  ry = 4;
			std::string grid = build_grid(av, rx, ry);
			os << ',' << json_int("grid_radius_x", rx);
			os << ',' << json_int("grid_radius_y", ry);
			// Top-left tile of the grid, so grid cell [row][col] maps to the
			// absolute tile (origin_tx+col, origin_ty+row). Avatar is at center.
			os << ',' << json_int("grid_origin_tx", av->get_tile().tx - rx);
			os << ',' << json_int("grid_origin_ty", av->get_tile().ty - ry);
			os << ',' << json_str("grid_legend",
					"@=you C=companion &=person b=lootable-body x=corpse(empty) "
					"n=container *=item E=exit/route(gate/stairs) ~=water "
					"==fence/barrier +=closed_door /=open_door .=walkable #=blocked; "
					"north=up east=right. Cell [row][col] is tile "
					"(grid_origin_tx+col, grid_origin_ty+row).");
			os << ',' << json_str("grid", grid);
		}

		// Nearby doors (with open/closed state) so the agent can plan routes
		// and know when to "open" a door to pass through.
		os << ',' << "\"doors\":[";
		if (av) {
			Game_object_vector objs;
			const Tile_coord   at = av->get_tile();
			Game_object::find_nearby(objs, at, -1, 12, 128);
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

		// Ambient speech: floating text shown over characters/objects outside a
		// formal conversation (e.g. a cat's "Meeow", townsfolk barks, warnings).
		os << ',' << "\"ambient_speech\":[";
		{
			Effects_manager* eff = gwin->get_effects();
			bool             afirst = true;
			if (eff) {
				for (auto& [txt, speaker] : eff->get_active_texts()) {
					if (txt.empty()) {
						continue;
					}
					// Only report text that a CHARACTER is saying (a bark).
					// Exult also shows floating text for other reasons - most
					// notably an item's NAME when you single-click it. Those
					// have a non-actor item as their owner (or the text equals
					// the item's own name), and are NOT speech. Skip them so
					// clicking an item (e.g. "Gargoyle jewelry") is not
					// misreported to the agent as something being said.
					Actor* act = speaker ? speaker->as_actor() : nullptr;
					if (!act) {
						continue;    // not a character talking -> not speech
					}
					// Guard against an actor's own name label (rare) being
					// treated as speech.
					if (txt == speaker->get_name()) {
						continue;
					}
					std::string who = act->get_name();
					if (!afirst) {
						os << ',';
					}
					afirst = false;
					os << '{' << json_str("who", who) << ',' << json_str("said", txt) << '}';
				}
			}
		}
		os << ']';

		// Active numeric-input prompt (slider + checkmark), e.g. "how many?".
		Slider_gump* sg = Slider_gump::get_active();
		os << ',' << json_bool("number_prompt", sg != nullptr);
		if (sg) {
			os << ',' << json_int("number_min", sg->get_min());
			os << ',' << json_int("number_max", sg->get_max());
			os << ',' << json_int("number_current", sg->get_val());
		}

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

		if (type == "wait_until") {
			// Advance the game clock to a target hour (0-23), e.g. wait for
			// morning so sleeping NPCs wake. Only advances forward (to today or
			// tomorrow at that hour). Default target = 7 (morning).
			Game_clock* clk = gwin->get_clock();
			if (!clk) {
				return "{\"ok\":false,\"error\":\"no clock\"}";
			}
			long target = 7;
			get_int(action_json, "hour", target);
			target = (target % 24 + 24) % 24;
			const int cur = clk->get_hour();
			int advance = static_cast<int>(target) - cur;
			if (advance <= 0) {
				advance += 24;    // next day
			}
			clk->set_hour(static_cast<int>(target));
			clk->set_minute(0);
			clk->set_palette();    // refresh day/night lighting
			// Re-evaluate NPC schedules for the new hour so sleepers wake and
			// townsfolk move to their daytime activities (setting the clock
			// alone does not transition schedules).
			gwin->schedule_npcs(static_cast<int>(target));
			return std::string("{\"ok\":true,\"did\":\"wait_until\",")
				   + json_int("hour", static_cast<int>(target)) + ","
				   + json_int("advanced_hours", advance) + "}";
		}

		if (type == "save") {
			// Persist the game (writes gamedat / quicksave) so progress is not
			// lost when the instance is closed.
			try {
				gwin->write();
			} catch (...) {
				return "{\"ok\":false,\"error\":\"save failed\"}";
			}
			return "{\"ok\":true,\"did\":\"save\"}";
		}

		if (type == "stop") {
			gwin->stop_actor();
			return "{\"ok\":true,\"did\":\"stop\"}";
		}

		if (type == "key" || type == "press_key") {
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

		if (type == "set_combat_mode") {
			// Set how the avatar (and party) fight when in combat. Names map to
			// Exult's Attack_mode enum.
			string mode;
			get_string(action_json, "mode", mode);
			std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
			static const std::map<std::string, Actor::Attack_mode> modes = {
				{"nearest", Actor::nearest}, {"weakest", Actor::weakest},
				{"strongest", Actor::strongest}, {"berserk", Actor::berserk},
				{"protect", Actor::protect}, {"defend", Actor::defend},
				{"flank", Actor::flank}, {"flee", Actor::flee},
				{"random", Actor::random}, {"manual", Actor::manual}};
			auto it = modes.find(mode);
			if (it == modes.end()) {
				return "{\"ok\":false,\"error\":\"unknown combat mode\"}";
			}
			Actor* cav = gwin->get_main_actor();
			if (cav) {
				cav->set_attack_mode(it->second, true);
			}
			// Apply to party members too so the whole group fights consistently.
			Party_manager* pm = gwin->get_party_man();
			if (pm) {
				for (int i = 0; i < pm->get_count(); ++i) {
					Game_object* m = gwin->get_npc(pm->get_member(i));
					if (Actor* ma = m ? m->as_actor() : nullptr) {
						ma->set_attack_mode(it->second, true);
					}
				}
			}
			return "{\"ok\":true,\"did\":\"set_combat_mode\"," + json_str("mode", mode) + "}";
		}

		if (type == "combat_pause") {
			ActionCombatPause(nullptr);
			return "{\"ok\":true,\"did\":\"combat_pause\"}";
		}

		if (type == "inventory") {
			// Report what the avatar is wearing (per slot) and carrying,
			// rather than just opening the (headless-useless) gump.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			static const struct {
				int         slot;
				const char* name;
			} slots[] = {
					{head, "head"},     {torso, "torso"},   {legs, "legs"},
					{feet, "feet"},     {rhand, "weapon"},  {lhand, "shield/hand"},
					{belt, "belt"},     {amulet, "amulet"}, {cloak, "cloak"},
					{gloves, "gloves"}, {lfinger, "ring"},  {backpack, "backpack"}};
			std::ostringstream inv;
			inv << "{\"ok\":true,\"did\":\"inventory\",\"worn\":{";
			bool wfirst = true;
			for (const auto& s : slots) {
				Game_object* it = av->get_readied(s.slot);
				if (it && !it->get_name().empty()) {
					if (!wfirst) {
						inv << ',';
					}
					wfirst = false;
					inv << '"' << s.name << "\":\"" << json_escape(it->get_name()) << '"';
				}
			}
			inv << "},\"carried\":[";
			// Everything in the backpack/held containers.
			bool  cfirst = true;
			int   ccount = 0;
			Container_game_object* pack = av->get_readied(backpack)
					? av->get_readied(backpack)->as_container()
					: nullptr;
			std::vector<Container_game_object*> stack;
			if (pack) {
				stack.push_back(pack);
			}
			while (!stack.empty() && ccount < 40) {
				Container_game_object* c = stack.back();
				stack.pop_back();
				Object_iterator it(c->get_objects());
				Game_object* inner;
				while ((inner = it.get_next()) != nullptr && ccount < 40) {
					if (!inner->get_name().empty()) {
						if (!cfirst) {
							inv << ',';
						}
						cfirst = false;
						inv << '"' << json_escape(inner->get_name()) << '"';
						++ccount;
					}
					if (Container_game_object* ic = inner->as_container()) {
						stack.push_back(ic);
					}
				}
			}
			inv << "]}";
			return inv.str();
		}

		if (type == "close") {
			// Close any open gump (e.g. a searched body/container) - equivalent
			// to pressing the checkmark/close on the gump.
			Gump_manager* gm = gwin->get_gump_man();
			if (gm && gm->showing_gumps(true)) {
				gm->close_all_gumps();
				return "{\"ok\":true,\"did\":\"close\"}";
			}
			return "{\"ok\":true,\"did\":\"close\",\"note\":\"nothing open\"}";
		}

		if (type == "equip") {
			// Ready (wear/wield) a named item. Looks in the avatar's inventory
			// and nearby containers; add_readied() auto-places it in the correct
			// slot based on the item's ready-type.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			string want;
			if (!get_string(action_json, "name", want)) {
				return "{\"ok\":false,\"error\":\"missing name\"}";
			}
			std::string wlow = want;
			std::transform(wlow.begin(), wlow.end(), wlow.begin(), ::tolower);
			auto matches = [&](Game_object* o) {
				std::string l = o->get_name();
				std::transform(l.begin(), l.end(), l.begin(), ::tolower);
				return !l.empty() && l.find(wlow) != std::string::npos;
			};
			// Search: avatar's own inventory first, then nearby containers.
			std::vector<Container_game_object*> roots;
			roots.push_back(av);    // Actor is-a container
			const Tile_coord   at = av->get_tile();
			Game_object_vector nearobjs;
			Game_object::find_nearby(nearobjs, at, -1, 3, 128);
			for (Game_object* o : nearobjs) {
				if (o && o != av) {
					if (Container_game_object* c = o->as_container()) {
						roots.push_back(c);
					}
				}
			}
			Game_object* found = nullptr;
			for (Container_game_object* root : roots) {
				std::vector<Container_game_object*> stk{root};
				while (!stk.empty() && !found) {
					Container_game_object* c = stk.back();
					stk.pop_back();
					Object_iterator it(c->get_objects());
					Game_object* inner;
					while ((inner = it.get_next()) != nullptr) {
						if (matches(inner)) {
							found = inner;
							break;
						}
						if (Container_game_object* ic = inner->as_container()) {
							stk.push_back(ic);
						}
					}
				}
				if (found) {
					break;
				}
			}
			if (!found) {
				return "{\"ok\":false,\"error\":\"item not found to equip\"}";
			}
			const std::string nm = found->get_name();
			// Detach and ready it. Try each real equip slot; add_readied()
			// validates that the item fits, so the first accepted slot is the
			// correct one. Fall back to carrying it if none accept.
			Game_object_shared keep;
			found->remove_this(&keep);
			static const int try_slots[] = {rhand, lhand, head, torso, legs, feet,
											belt, amulet, cloak, gloves, lfinger,
											rfinger, quiver, backpack};
			for (int s : try_slots) {
				if (!av->get_readied(s) && av->add_readied(found, s, false, false)) {
					return "{\"ok\":true,\"did\":\"equip\",\"item\":\"" + json_escape(nm)
						   + "\",\"slot\":" + std::to_string(s) + "}";
				}
			}
			if (av->add(found, false, true)) {
				return "{\"ok\":false,\"error\":\"could not wear '" + json_escape(nm)
					   + "'; kept in pack\"}";
			}
			found->set_invalid();
			found->move(at.tx, at.ty, at.tz);
			return "{\"ok\":false,\"error\":\"could not equip '" + json_escape(nm) + "'\"}";
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

		if (type == "search") {
			// Open (activate) the nearest body or unlocked container so its
			// contents become accessible - this is how you loot a murder
			// victim or a chest.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			const Tile_coord   at = av->get_tile();
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, 4, 128);
			Game_object* best   = nullptr;
			int          best_d = 1 << 30;
			for (Game_object* obj : objs) {
				if (!obj) {
					continue;
				}
				const Shape_info& info = obj->get_info();
				const bool        is_container
						= info.get_shape_class() == Shape_info::container;
				// A murder victim is often a DEAD ACTOR (corpse), not a
				// body-shape object. Include dead actors so the agent can loot
				// a slain NPC just like a body/chest.
				Actor* act = obj->as_actor();
				const bool is_dead_actor = act && act->is_dead();
				if (!info.is_body_shape() && !is_container && !is_dead_actor) {
					continue;
				}
				const Tile_coord ot = obj->get_tile();
				const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
				if (d < best_d) {
					best_d = d;
					best   = obj;
				}
			}
			// Also scan nearby NPCs for a dead one, since dead actors may be
			// tracked in the actor list rather than the object list.
			{
				std::vector<Actor*> npcs;
				gwin->get_nearby_npcs(npcs);
				for (Actor* npc : npcs) {
					if (!npc || npc == av || !npc->is_dead()) {
						continue;
					}
					const Tile_coord ot = npc->get_tile();
					const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
					if (d <= 4 && d < best_d) {
						best_d = d;
						best   = npc;
					}
				}
			}
			if (!best) {
				return "{\"ok\":false,\"error\":\"no body or container nearby\"}";
			}
			const std::string nm = best->get_name();
			best->activate();    // opens the body/container (shows the gump)
			// LOOT it: transfer takeable contents into the avatar's inventory,
			// like a person emptying a bag/chest/corpse. This is what makes
			// "search" actually useful - just opening the gump left the gold,
			// food, torches etc. sitting inside. We take everything that is
			// freely takeable; a non-container corpse simply has nothing.
			Container_game_object* cont = best->as_container();
			std::string took;
			int         took_n = 0;
			if (cont) {
				Game_object_vector contents;
				cont->get_objects(contents, c_any_shapenum, c_any_qual, c_any_framenum);
				for (Game_object* it : contents) {
					if (!it) {
						continue;
					}
					const std::string inm = it->get_name();
					Game_object_shared keep;
					it->remove_this(&keep);
					if (av->add(it, false, true)) {
						if (took_n < 12) {
							if (took_n) {
								took += ", ";
							}
							took += inm;
						}
						++took_n;
					} else {
						// Couldn't carry it (too heavy/full) - put it back.
						cont->add(it, true);
					}
				}
			}
			if (took_n > 0) {
				return "{\"ok\":true,\"did\":\"search\",\"target\":\""
					   + json_escape(nm) + "\",\"looted\":\"" + json_escape(took)
					   + "\",\"count\":" + std::to_string(took_n) + "}";
			}
			return "{\"ok\":true,\"did\":\"search\",\"target\":\""
				   + json_escape(nm) + "\",\"looted\":\"\",\"empty\":true}";
		}

		if (type == "read") {
			// Read the nearest SIGN / readable object (a human double-clicks it).
			// Activating a sign fires its usecode, which shows floating text; we
			// capture that text and return it. Optional "name" to disambiguate.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			string want;
			get_string(action_json, "name", want);
			std::string wlow = want;
			std::transform(wlow.begin(), wlow.end(), wlow.begin(), ::tolower);
			const Tile_coord   at = av->get_tile();
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, 4, 128);
			Game_object* best   = nullptr;
			int          best_d = 1 << 30;
			for (Game_object* obj : objs) {
				if (!obj || obj->as_actor()) {
					continue;
				}
				std::string nm = obj->get_name();
				if (nm.empty()) {
					continue;
				}
				std::string nlow = nm;
				std::transform(nlow.begin(), nlow.end(), nlow.begin(), ::tolower);
				// A "read" target is a sign/plaque/marker by default, or any
				// object matching the requested name.
				const bool is_signish = nlow.find("sign") != std::string::npos
						|| nlow.find("plaque") != std::string::npos
						|| nlow.find("placard") != std::string::npos
						|| nlow.find("marker") != std::string::npos
						|| nlow.find("tombstone") != std::string::npos
						|| nlow.find("grave") != std::string::npos;
				if (!wlow.empty()) {
					if (nlow.find(wlow) == std::string::npos) {
						continue;
					}
				} else if (!is_signish) {
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
				return "{\"ok\":false,\"error\":\"no sign/readable object nearby\"}";
			}
			const std::string nm = best->get_name();
			// Reading a sign runs usecode that shows a MODAL text gump and waits
			// for a click (Get_click). We (a) clear the captured-text global,
			// (b) pre-inject a click so the modal dismisses itself immediately,
			// then (c) activate the sign. display_runes fills the global with
			// the sign's text as it builds the gump, so we can return it even
			// though the gump auto-closes. (A sign is a modal you click off.)
			extern std::string g_llm_last_sign_text;
			g_llm_last_sign_text.clear();
			// Inject a left mouse click so the modal Get_click returns at once.
			{
				SDL_Event ev = {};
				ev.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
				ev.button.button = SDL_BUTTON_LEFT;
				ev.button.x = 10;
				ev.button.y = 10;
				SDL_PushEvent(&ev);
				ev.type = SDL_EVENT_MOUSE_BUTTON_UP;
				SDL_PushEvent(&ev);
			}
			best->activate();    // shows the sign, caches text, click dismisses it
			const std::string text = g_llm_last_sign_text;
			return "{\"ok\":true,\"did\":\"read\",\"target\":\""
				   + json_escape(nm) + "\",\"text\":\"" + json_escape(text) + "\"}";
		}

		if (type == "pickup") {
			// Take the nearest takeable world object (optionally matching a
			// name) into the avatar's inventory.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			string want;
			get_string(action_json, "name", want);
			std::string wlow = want;
			std::transform(wlow.begin(), wlow.end(), wlow.begin(), ::tolower);

			const Tile_coord   at = av->get_tile();
			Game_object_vector objs;
			// Reach a bit further than before (was 3): items shown in the
			// observation (radius up to 12) were often just out of pickup
			// range, causing "no takeable object nearby" even though the agent
			// could see them. 6 tiles is a reasonable arm's reach for a tile
			// game and matches how the pathfinder gets you adjacent.
			Game_object::find_nearby(objs, at, -1, 6, 128);
			Game_object* best   = nullptr;
			int          best_d = 1 << 30;
			for (Game_object* obj : objs) {
				if (!obj || obj->as_actor()) {
					continue;
				}
				const Shape_info& info = obj->get_info();
				// Skip fixed scenery: doors, furniture-like, buildings.
				if (info.is_door()) {
					continue;
				}
				const std::string nm = obj->get_name();
				if (nm.empty()) {
					continue;
				}
				if (!wlow.empty()) {
					std::string nlow = nm;
					std::transform(nlow.begin(), nlow.end(), nlow.begin(), ::tolower);
					if (nlow.find(wlow) == std::string::npos) {
						continue;
					}
				}
				const Tile_coord ot = obj->get_tile();
				const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
				if (d < best_d) {
					best_d = d;
					best   = obj;
				}
			}
			if (!best) {
				return "{\"ok\":false,\"error\":\"no takeable object nearby\"}";
			}
			const std::string nm = best->get_name();
			// Detach from the world, keeping a shared ref alive, then add to
			// the avatar's inventory.
			Game_object_shared keep;
			best->remove_this(&keep);
			if (av->add(best, false, true)) {
				return "{\"ok\":true,\"did\":\"pickup\",\"item\":\"" + json_escape(nm) + "\"}";
			}
			// Couldn't carry it - drop it back where the avatar stands.
			best->set_invalid();
			best->move(at.tx, at.ty, at.tz);
			return "{\"ok\":false,\"error\":\"could not carry '" + json_escape(nm) + "'\"}";
		}

		if (type == "take") {
			// Take an item from a nearby container/body (searching recursively
			// through bags) into the avatar's inventory. Optional {name} picks
			// a specific item; otherwise takes the first item found.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			string want;
			get_string(action_json, "name", want);
			std::string wlow = want;
			std::transform(wlow.begin(), wlow.end(), wlow.begin(), ::tolower);
			const Tile_coord   at = av->get_tile();
			Game_object_vector objs;
			Game_object::find_nearby(objs, at, -1, 3, 128);
			// Gather all containers nearby (bodies, bags, chests).
			std::vector<Container_game_object*> conts;
			for (Game_object* obj : objs) {
				if (obj) {
					if (Container_game_object* c = obj->as_container()) {
						conts.push_back(c);
					}
				}
			}
			// Walk the container tree to find a matching item.
			Game_object* found = nullptr;
			for (Container_game_object* root : conts) {
				std::vector<Container_game_object*> stack{root};
				while (!stack.empty() && !found) {
					Container_game_object* c = stack.back();
					stack.pop_back();
					Object_iterator it(c->get_objects());
					Game_object* inner;
					while ((inner = it.get_next()) != nullptr) {
						const std::string inm = inner->get_name();
						bool match = !inm.empty();
						if (match && !wlow.empty()) {
							std::string l = inm;
							std::transform(l.begin(), l.end(), l.begin(), ::tolower);
							match = l.find(wlow) != std::string::npos;
						}
						if (match) {
							found = inner;
							break;
						}
						if (Container_game_object* ic = inner->as_container()) {
							stack.push_back(ic);
						}
					}
				}
				if (found) {
					break;
				}
			}
			if (!found) {
				return "{\"ok\":false,\"error\":\"no such item in a nearby container\"}";
			}
			const std::string nm = found->get_name();
			Game_object_shared keep;
			found->remove_this(&keep);
			if (av->add(found, false, true)) {
				return "{\"ok\":true,\"did\":\"take\",\"item\":\"" + json_escape(nm) + "\"}";
			}
			// Couldn't carry it - drop at feet so it isn't lost.
			found->set_invalid();
			found->move(at.tx, at.ty, at.tz);
			return "{\"ok\":false,\"error\":\"could not carry '" + json_escape(nm) + "'\"}";
		}

		if (type == "goto") {
			// Pathfind (A*) to a destination: an explicit tile {tx,ty}, or the
			// nearest NPC/object matching {name}.  Routes around walls and
			// through doorways automatically.
			Actor* av = gwin->get_main_actor();
			if (!av) {
				return "{\"ok\":false,\"error\":\"no avatar\"}";
			}
			const Tile_coord at = av->get_tile();
			long tx = -1;
			long ty = -1;
			Tile_coord dest(0, 0, at.tz);
			bool have_dest = false;
			bool tz_was_explicit = false;
			if (get_int(action_json, "tx", tx) && get_int(action_json, "ty", ty)) {
				// Accept an explicit target elevation (tz) if given, so the LLM
				// can disambiguate levels (ground vs wall-top vs underground).
				long tz_in = at.tz;
				tz_was_explicit = get_int(action_json, "tz", tz_in);
				dest      = Tile_coord(static_cast<int>(tx), static_cast<int>(ty),
									   static_cast<int>(tz_in));
				have_dest = true;
			} else {
				string want;
				if (get_string(action_json, "name", want)) {
					std::string wlow = want;
					std::transform(wlow.begin(), wlow.end(), wlow.begin(), ::tolower);
					// Search NPCs first, then objects.
					int best_d = 1 << 30;
					std::vector<Actor*> npcs;
					gwin->get_nearby_npcs(npcs);
					for (Actor* npc : npcs) {
						if (!npc || npc == av) {
							continue;
						}
						std::string nlow = npc->get_name();
						std::transform(nlow.begin(), nlow.end(), nlow.begin(), ::tolower);
						if (nlow.find(wlow) == std::string::npos) {
							continue;
						}
						const Tile_coord nt = npc->get_tile();
						const int d = std::abs(nt.tx - at.tx) + std::abs(nt.ty - at.ty);
						if (d < best_d) {
							best_d = d;
							dest = nt;
							have_dest = true;
						}
					}
					if (!have_dest) {
						Game_object_vector objs;
						Game_object::find_nearby(objs, at, -1, 18, 128);
						for (Game_object* obj : objs) {
							if (!obj || obj->as_actor() || obj->get_name().empty()) {
								continue;
							}
							std::string nlow = obj->get_name();
							std::transform(nlow.begin(), nlow.end(), nlow.begin(), ::tolower);
							if (nlow.find(wlow) == std::string::npos) {
								continue;
							}
							const Tile_coord ot = obj->get_tile();
							const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
							if (d < best_d) {
								best_d = d;
								dest = ot;
								have_dest = true;
							}
						}
					}
				}
			}
			if (!have_dest) {
				return "{\"ok\":false,\"error\":\"no destination (give tx/ty or a visible name)\"}";
			}
			// Z-LAYER RESOLUTION: only when the caller did NOT give an explicit
			// tz. goto defaults the destination to the avatar's current Z; a
			// stairs/wall-top tile is only walkable at a higher Z, so targeting
			// it at ground Z paths to "under the stairs". If the dest tile is
			// blocked at the current Z but a standable surface exists higher up,
			// retarget to that Z so the pathfinder climbs. (If the LLM gave an
			// explicit tz we TRUST it and skip this heuristic.)
			if (!tz_was_explicit) {
				Game_map* gmap = gwin->get_map();
				Map_chunk* dchunk = gmap ? gmap->get_chunk(
						dest.tx / c_tiles_per_chunk, dest.ty / c_tiles_per_chunk) : nullptr;
				if (dchunk) {
					dchunk->setup_cache();
					const int lx = dest.tx % c_tiles_per_chunk;
					const int ly = dest.ty % c_tiles_per_chunk;
					int new_lift = dest.tz;
					const bool blk = dchunk->is_blocked(
							2, dest.tz, lx, ly, new_lift, av->get_type_flags(),
							1 /*max_drop*/, 6 /*max_rise: allow climbing stairs*/);
					if (blk) {
						// Blocked at this Z; probe upward for a standable surface.
						for (int z = dest.tz + 1; z <= dest.tz + 6; ++z) {
							int nl = z;
							if (!dchunk->is_blocked(2, z, lx, ly, nl,
									av->get_type_flags(), 1, 0)) {
								dest.tz = z;
								break;
							}
						}
					} else if (new_lift != dest.tz) {
						dest.tz = new_lift;   // stepping up onto a surface
					}
				}
			}
			long speed = 200;
			get_int(action_json, "speed", speed);
			if (av->walk_path_to_tile(dest, static_cast<int>(speed))) {
				return "{\"ok\":true,\"did\":\"goto\"," + json_int("tx", dest.tx) + ","
					   + json_int("ty", dest.ty) + "}";
			}
			// Path failed - a closed door may be blocking. Open the nearest
			// closed door and retry once (the pathfinder also auto-opens doors
			// it walks into, but only if it found a path in the first place).
			{
				Game_object_vector nd;
				Game_object::find_nearby(nd, at, -1, 6, 128);
				Game_object* door = nullptr;
				int          bd   = 1 << 30;
				for (Game_object* obj : nd) {
					if (!obj || !obj->get_info().is_door()) {
						continue;
					}
					if ((obj->get_framenum() % 4) >= 2) {
						continue;    // already open
					}
					const Tile_coord ot = obj->get_tile();
					const int d = std::abs(ot.tx - at.tx) + std::abs(ot.ty - at.ty);
					if (d < bd) {
						bd = d;
						door = obj;
					}
				}
				if (door) {
					door->activate();    // open it
					if (av->walk_path_to_tile(dest, static_cast<int>(speed))) {
						return "{\"ok\":true,\"did\":\"goto\",\"opened_door\":true,"
							   + json_int("tx", dest.tx) + "," + json_int("ty", dest.ty) + "}";
					}
				}
			}
			// Pathfinder gave up on the exact destination (too far / walls in
			// the way). A human sees the 2D map and walks to the nearest
			// reachable spot toward the target, then continues. Emulate that:
			// try A* to intermediate tiles along the line to the target,
			// progressively closer to us, and go to the first REACHABLE one.
			{
				const int dx = dest.tx - at.tx;
				const int dy = dest.ty - at.ty;
				const int dist = std::max(std::abs(dx), std::abs(dy));
				// Try waypoints at ~90%, 75%, 60%, ... of the way, plus small
				// lateral offsets, so we route around an obstacle rather than
				// give up. First reachable waypoint wins.
				static const double fracs[] = {0.85, 0.7, 0.55, 0.4, 0.25, 0.15};
				for (double f : fracs) {
					const int wx = at.tx + static_cast<int>(dx * f);
					const int wy = at.ty + static_cast<int>(dy * f);
					// Try the point and a few lateral nudges (to slip around
					// a wall corner).
					const int off[][2] = {{0, 0}, {2, 0}, {-2, 0}, {0, 2},
										  {0, -2}, {3, 3}, {-3, 3}, {3, -3}, {-3, -3}};
					for (auto& o : off) {
						Tile_coord wp(
								(wx + o[0] + c_num_tiles) % c_num_tiles,
								(wy + o[1] + c_num_tiles) % c_num_tiles, at.tz);
						if (wp.tx == at.tx && wp.ty == at.ty) {
							continue;
						}
						if (av->walk_path_to_tile(wp, static_cast<int>(speed))) {
							return "{\"ok\":true,\"did\":\"goto\",\"partial\":true,"
								   + json_int("tx", wp.tx) + "," + json_int("ty", wp.ty)
								   + "," + json_int("toward_tx", dest.tx) + ","
								   + json_int("toward_ty", dest.ty) + "}";
						}
					}
					(void)dist;
				}
			}
			// Last resort: a single walk-step toward the target if the adjacent
			// tile is free (handles the very-close case).
			{
				Game_map* gmap = gwin->get_map();
				const int ddx  = (dest.tx > at.tx) - (dest.tx < at.tx);
				const int ddy  = (dest.ty > at.ty) - (dest.ty < at.ty);
				if ((ddx || ddy) && gmap) {
					const Tile_coord step(
							(at.tx + ddx + c_num_tiles) % c_num_tiles,
							(at.ty + ddy + c_num_tiles) % c_num_tiles, at.tz);
					if (!gmap->is_tile_occupied(step)) {
						const int w   = gwin->get_width();
						const int h   = gwin->get_height();
						const int sx  = w / 2 + ddx * 40;
						const int sy  = h / 2 + ddy * 40;
						gwin->start_actor(sx, sy, static_cast<int>(speed));
						return "{\"ok\":true,\"did\":\"goto\",\"stepped\":true,"
							   + json_int("tx", dest.tx) + "," + json_int("ty", dest.ty) + "}";
					}
				}
			}
			return "{\"ok\":false,\"error\":\"no path to destination\"}";
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
			Game_object::find_nearby(objs, at, -1, 4, 128);
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

		if (type == "set_number") {
			Slider_gump* sg = Slider_gump::get_active();
			if (!sg) {
				return "{\"ok\":false,\"error\":\"no number prompt active\"}";
			}
			long v = 0;
			if (!get_int(action_json, "value", v)) {
				return "{\"ok\":false,\"error\":\"missing value\"}";
			}
			sg->set_value_and_confirm(static_cast<int>(v));
			return "{\"ok\":true,\"did\":\"set_number\"," + json_int("value", v) + "}";
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
			// Detect whether the adjacent tile in this direction is blocked
			// (wall/water/obstacle), so the agent gets an explicit "bumped"
			// signal instead of silently walking into a wall.
			bool blocked = false;
			{
				Game_map* gmap = gwin->get_map();
				Actor*    av   = gwin->get_main_actor();
				if (gmap && av) {
					const Tile_coord me = av->get_tile();
					int ddx = 0;
					int ddy = 0;
					if (dir.find('n') != string::npos) {
						ddy = -1;
					}
					if (dir.find('s') != string::npos) {
						ddy = 1;
					}
					if (dir.find('e') != string::npos) {
						ddx = 1;
					}
					if (dir.find('w') != string::npos) {
						ddx = -1;
					}
					const Tile_coord adj(
							(me.tx + ddx + c_num_tiles) % c_num_tiles,
							(me.ty + ddy + c_num_tiles) % c_num_tiles, me.tz);
					blocked = gmap->is_tile_occupied(adj);
				}
			}
			gwin->start_actor(tx, ty, static_cast<int>(speed));
			return string("{\"ok\":true,\"did\":\"move\",\"dir\":\"") + json_escape(dir)
				   + "\"," + json_bool("blocked", blocked) + "}";
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
		if (cmd == "screenshot") {
			return screenshot();
		}
		return "{\"ok\":false,\"error\":\"unknown cmd\"}";
	}

	// Capture the current game screen to a fixed file and return its path, so
	// an external (more capable) agent can visually inspect what the engine is
	// actually rendering and compare it against the JSON observation. Overwrites
	// the same file each call.
	string screenshot() {
		Game_window* gwin = Game_window::get_instance();
		if (!gwin || !gwin->get_win()) {
			return "{\"ok\":false,\"error\":\"no window\"}";
		}
		const string dir = get_system_path("<SAVEGAME>");
		const string path = dir + "/agent_shot.png";
		SDL_IOStream* dst = SDL_IOFromFile(path.c_str(), "wb");
		if (!dst) {
			return "{\"ok\":false,\"error\":\"cannot open output file\"}";
		}
		const bool ok = gwin->get_win()->screenshot(dst, false);
		// screenshot() closes dst via SDL_SaveBMP/IMG path; guard anyway.
		if (ok) {
			return "{\"ok\":true,\"did\":\"screenshot\"," + json_str("path", path) + "}";
		}
		return "{\"ok\":false,\"error\":\"screenshot failed\"}";
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

// Global-scope shim so usecode (intrinsics.cc) can stash the most recent sign
// text without needing the LLM_agent namespace.
void LLM_agent_set_last_sign_text(const std::string& text) {
	LLM_agent::g_llm_last_sign_text = text;
}

#endif /* USE_LLM_AGENT */
