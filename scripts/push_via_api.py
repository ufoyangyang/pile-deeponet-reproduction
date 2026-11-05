"""Push the local git tree to GitHub through the REST Git Data API.

The network here blocks `git push` over https (github.com:443 resets) while
api.github.com works, so the snapshot is uploaded blob-by-blob and assembled
into a single commit on the remote.

The Git Data API rejects ref operations on an empty repository, so if the remote
has no branch yet we seed it with one file through the Contents API first and
then force-update the ref.
"""

import base64
import json
import subprocess
import sys
import urllib.request

REPO = "ufoyangyang/pile-deeponet-reproduction"
TOKEN = subprocess.check_output(["gh", "auth", "token"], text=True, encoding="utf-8").strip()


def api(method, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        "https://api.github.com" + path, method=method, data=body,
        headers={"Authorization": "Bearer " + TOKEN,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "pile-deeponet-push"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt else {}


def head_sha():
    try:
        return api("GET", f"/repos/{REPO}/git/ref/heads/main")["object"]["sha"]
    except Exception:
        return None


def seed():
    """Create the very first commit (the Git Data API needs a non-empty repo)."""
    body = b"# pile-deeponet-reproduction\n\nseeding commit, content follows\n"
    res = api("PUT", f"/repos/{REPO}/contents/README.md", {
        "message": "seed", "content": base64.b64encode(body).decode(),
    })
    return res["commit"]["sha"]


def main():
    parent = head_sha()
    if parent is None:
        parent = seed()
        print("seeded empty repo, parent", parent)

    raw = subprocess.check_output(["git", "ls-files", "-z"])
    files = [f for f in raw.decode("utf-8").split(chr(0)) if f]
    print(f"uploading {len(files)} files")
    tree = []
    for i, f in enumerate(files):
        blob = api("POST", f"/repos/{REPO}/git/blobs",
                   {"content": base64.b64encode(open(f, "rb").read()).decode(),
                    "encoding": "base64"})
        tree.append({"path": f, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if i % 10 == 0:
            print(f"  {i}/{len(files)} {f}")

    root = api("POST", f"/repos/{REPO}/git/trees", {"tree": tree})
    msg = subprocess.check_output(["git", "log", "-1", "--pretty=%s"], encoding="utf-8").strip()
    commit = api("POST", f"/repos/{REPO}/git/commits",
                 {"message": msg, "tree": root["sha"], "parents": [parent]})
    print("commit", commit["sha"])
    api("PATCH", f"/repos/{REPO}/git/refs/heads/main",
        {"sha": commit["sha"], "force": True})
    api("PATCH", f"/repos/{REPO}", {"default_branch": "main"})
    print("done:", f"https://github.com/{REPO}")


if __name__ == "__main__":
    sys.exit(main())
