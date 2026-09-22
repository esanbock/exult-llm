"""play.py - one-shot manual bridge client for hands-on testing.

Usage:
  python play.py observe            # raw observation JSON (pretty)
  python play.py observe --compact  # observation minus the big grid
  python play.py act '<json>'       # send an action, print result
  python play.py talk <name>        # start a conversation
  python play.py raw '<json>'       # send any raw command dict

Sends exactly one JSON line to the Exult bridge (127.0.0.1:45999) and prints
the one reply. Separate from driver.py so it doesn't disturb its connection.
"""
import json
import socket
import sys

HOST, PORT = "127.0.0.1", 45999


def send(obj):
    s = socket.create_connection((HOST, PORT), timeout=30)
    try:
        s.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        buf = b""
        s.settimeout(30)
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    finally:
        s.close()


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: observe|act|talk|raw ...")
        return
    cmd = args[0]
    if cmd == "observe":
        r = send({"cmd": "observe"})
        if "--compact" in args:
            r.pop("grid", None)
        print(json.dumps(r, indent=2))
    elif cmd == "act":
        print(json.dumps(send({"cmd": "act", "action": json.loads(args[1])}), indent=2))
    elif cmd == "talk":
        print(json.dumps(send({"cmd": "talk", "name": args[1]}), indent=2))
    elif cmd == "raw":
        print(json.dumps(send(json.loads(args[1])), indent=2))
    else:
        print("unknown:", cmd)


if __name__ == "__main__":
    main()
