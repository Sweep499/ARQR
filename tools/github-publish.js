// Publishes a file to a GitHub repository from the browser, using the GitHub REST API and a personal access
// token the person pastes in (a fine-grained token limited to one repository with "Contents: read and write").
// Nothing here stores the token; the caller decides what to remember.

const API = 'https://api.github.com';

export class PublishError extends Error {
  constructor(kind, message) { super(message); this.kind = kind; }   // kind: auth | notfound | changed | conflict | network | other
}

const utf8ToB64 = (text) => {
  const bytes = new TextEncoder().encode(text);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
};
const b64ToUtf8 = (b64) => {
  const bin = atob(b64.replace(/\s/g, ''));
  return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
};

async function call(url, token, init = {}) {
  let res;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      },
    });
  } catch (e) {
    throw new PublishError('network', 'Could not reach GitHub. Check the connection and try again.');
  }
  return res;
}

// Replaces `path` on `branch` with `text`.
//   baseText  what the person's copy was based on; if GitHub's file differs from it, this stops with kind
//             "changed" (unless force) so someone else's edit is not overwritten by accident.
// Resolves { status: 'unchanged' } or { status: 'committed', commitUrl, commitSha }.
export async function publishFile({ token, repo, branch, path, text, message, baseText = null, force = false }) {
  if (!token) throw new PublishError('auth', 'Paste a GitHub token first.');
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new PublishError('other', 'The repository should look like owner/name.');
  const url = `${API}/repos/${repo}/contents/${path.split('/').map(encodeURIComponent).join('/')}`;

  const get = await call(`${url}?ref=${encodeURIComponent(branch)}`, token);
  if (get.status === 401) throw new PublishError('auth', 'GitHub did not accept the token. It may be wrong, expired or revoked.');
  if (get.status === 404) throw new PublishError('notfound', `Could not find ${path} on ${repo} (${branch}), or the token has no access to that repository.`);
  if (!get.ok) throw new PublishError('other', `GitHub answered ${get.status} when reading ${path}.`);
  const remote = await get.json();
  const remoteText = b64ToUtf8(remote.content || '');

  if (remoteText === text) return { status: 'unchanged' };
  if (baseText !== null && !force && remoteText !== baseText) {
    throw new PublishError('changed', 'The copy on GitHub is different from the one this page started from, so someone may have changed it since. Publishing would replace their version.');
  }

  const put = await call(url, token, { method: 'PUT', body: JSON.stringify({ message, content: utf8ToB64(text), sha: remote.sha, branch }) });
  if (put.status === 401) throw new PublishError('auth', 'GitHub did not accept the token.');
  if (put.status === 403 || put.status === 404) throw new PublishError('auth', 'The token can read this repository but not write to it. Give it "Contents: read and write".');
  if (put.status === 409 || put.status === 422) throw new PublishError('conflict', 'The file changed on GitHub while publishing. Try again.');
  if (!put.ok) throw new PublishError('other', `GitHub answered ${put.status} when writing ${path}.`);
  const done = await put.json();
  return { status: 'committed', commitUrl: done.commit?.html_url, commitSha: done.commit?.sha };
}
