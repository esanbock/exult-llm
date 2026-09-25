"""
twitch_bridge.py - Let Twitch chat interview/steer the LLM agent, live on stream.

Twitch chat runs on IRC (irc.chat.twitch.tv:6667). This bridge connects there
with no third-party dependencies (stdlib sockets only), listens for viewer
commands, and pipes them to the running agent driver via small queue files that
the driver already consumes:

  !ask  <question>   -> appended to ask_queue.txt  (driver interviews the agent,
                        out-of-band; does NOT affect gameplay)
  !hint <text>       -> appended to hint.txt        (driver steers the agent)

When the driver answers a question it writes Q&A to agent_answer.txt; this bridge
watches that file and:
  * posts the answer back to Twitch chat, and
  * writes it to obs_answer.txt for an OBS "Text (GDI+)" source to display on
    stream (point the source at that file with "read from file" enabled).

SETUP (no secrets in code - use environment variables):
  set TWITCH_OAUTH=oauth:xxxxxxxxxxxxxxxxxxxxxxxxxxxx   (get from
      https://twitchapps.com/tmi/ or your own app token; needs chat:read+chat:edit)
  set TWITCH_NICK=your_bot_or_channel_login
  set TWITCH_CHANNEL=your_channel_login   (the channel to join, lowercase)
Then:  python twitch_bridge.py

Rate/safety: viewer questions are length-capped and stripped of newlines; the
bridge only ever writes to the queue files - it cannot execute anything else.
"""
import os
import re
import socket
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ASK_QUEUE = os.path.join(HERE, "ask_queue.txt")
HINT_FILE = os.path.join(HERE, "hint.txt")
ANSWER_FILE = os.path.join(HERE, "agent_answer.txt")
OBS_ANSWER = os.path.join(HERE, "obs_answer.txt")

HOST = "irc.chat.twitch.tv"
PORT = 6667
MAX_MSG = 200            # cap viewer message length fed to the agent
ALLOW_HINTS = os.environ.get("TWITCH_ALLOW_HINTS", "1") == "1"


def _append_line(path: str, text: str) -> None:
    """Append a single sanitized line to a queue file (create if missing)."""
    line = re.sub(r"\s+", " ", text).strip()[:MAX_MSG]
    if not line:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_answer_mtime():
    try:
        return os.path.getmtime(ANSWER_FILE)
    except OSError:
        return 0


class TwitchBridge:
    def __init__(self, oauth, nick, channel):
        self.oauth = oauth
        self.nick = nick.lower()
        self.channel = "#" + channel.lower().lstrip("#")
        self.sock = None
        self._last_answer_mtime = _read_answer_mtime()

    def connect(self):
        self.sock = socket.socket()
        self.sock.settimeout(1.0)
        self.sock.connect((HOST, PORT))
        self._send_raw(f"PASS {self.oauth}")
        self._send_raw(f"NICK {self.nick}")
        self._send_raw(f"JOIN {self.channel}")
        print(f"[twitch] connected as {self.nick}, joined {self.channel}")
        self.chat(f"Agent bridge online. Use !ask <question> to interview the AI"
                  + (" or !hint <tip> to help it." if ALLOW_HINTS else "."))

    def _send_raw(self, line: str):
        self.sock.sendall((line + "\r\n").encode("utf-8"))

    def chat(self, msg: str):
        """Post a message to the channel (Twitch caps ~500 chars)."""
        msg = msg.replace("\r", " ").replace("\n", " ")[:480]
        try:
            self._send_raw(f"PRIVMSG {self.channel} :{msg}")
        except OSError:
            pass

    def _handle_privmsg(self, user: str, text: str):
        t = text.strip()
        low = t.lower()
        # Unified channel: !ask and !hint both go to the agent, which decides for
        # itself whether the message is a question to answer or guidance to act
        # on. (!hint kept as a friendly alias.)
        msg = None
        if low.startswith("!ask"):
            msg = t[4:].strip()
        elif low.startswith("!hint"):
            msg = t[5:].strip()
        if msg:
            _append_line(ASK_QUEUE, msg)
            print(f"[twitch] {user}: {msg}")
            # No ack posted - only the AI's actual answer goes to chat.

    def _poll_answer(self):
        """If the driver wrote a new answer, post it to chat + OBS file."""
        m = _read_answer_mtime()
        if m and m != self._last_answer_mtime:
            self._last_answer_mtime = m
            try:
                qa = open(ANSWER_FILE, encoding="utf-8").read().strip()
            except OSError:
                return
            if not qa:
                return
            # Write for OBS text source.
            try:
                with open(OBS_ANSWER, "w", encoding="utf-8") as f:
                    f.write(qa)
            except OSError:
                pass
            # Post to chat (split Q / A).
            for part in qa.split("\n"):
                if part.strip():
                    self.chat("AI " + part.strip())
                    time.sleep(1.2)   # gentle pacing to avoid Twitch rate limits

    def run(self):
        self.connect()
        buf = ""
        while True:
            try:
                data = self.sock.recv(4096).decode("utf-8", "ignore")
                if data:
                    buf += data
                    while "\r\n" in buf:
                        line, buf = buf.split("\r\n", 1)
                        if line.startswith("PING"):
                            self._send_raw("PONG :tmi.twitch.tv")
                            continue
                        # :user!user@user.tmi.twitch.tv PRIVMSG #chan :message
                        m = re.match(r"^:(\w+)!.* PRIVMSG #\S+ :(.*)$", line)
                        if m:
                            self._handle_privmsg(m.group(1), m.group(2))
            except socket.timeout:
                pass
            except OSError as e:
                print(f"[twitch] socket error: {e}; reconnecting in 5s")
                time.sleep(5)
                try:
                    self.connect()
                except OSError:
                    pass
            # Check for a new agent answer to relay, each loop.
            self._poll_answer()


def _load_env_file():
    """Optionally load credentials from a gitignored 'twitch.env' next to this
    file (KEY=VALUE lines), so they never live in code, chat, or git. Values
    already set in the real environment take precedence."""
    path = os.path.join(HERE, "twitch.env")
    if not os.path.isfile(path):
        return
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def main():
    _load_env_file()
    oauth = os.environ.get("TWITCH_OAUTH", "")
    nick = os.environ.get("TWITCH_NICK", "")
    channel = os.environ.get("TWITCH_CHANNEL", "")
    if not (oauth and nick and channel):
        print("Set TWITCH_OAUTH, TWITCH_NICK, TWITCH_CHANNEL - either as "
              "environment variables or in a gitignored 'twitch.env' file next "
              "to this script (KEY=VALUE lines). See the header for details.")
        return
    TwitchBridge(oauth, nick, channel).run()


if __name__ == "__main__":
    main()
