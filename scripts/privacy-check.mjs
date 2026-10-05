import { execFileSync } from "node:child_process";
import { readFileSync, lstatSync } from "node:fs";
import { extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const patterns = [
  [
    "private-key",
    /-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----/g,
  ],
  [
    "credential-token",
    /(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?:AKIA|ASIA)[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|hf_[A-Za-z0-9]{24,}|npm_[A-Za-z0-9]{30,})/g,
  ],
  [
    "personal-path",
    /(?:\/Users\/[^/\s]+\/|\/home\/[^/\s]+\/|[A-Z]:\\Users\\[^\\\s]+\\)/g,
  ],
  [
    "private-network",
    /\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b/g,
  ],
  ["personal-email", /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g],
  [
    "credential-assignment",
    /\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password|authorization)\b["']?\s*[:=]\s*["'][^"'\n]{8,}["']/gi,
  ],
  ["authenticated-url", /https?:\/\/[^\s/@]+:[^\s/@]+@/g],
  [
    "session-token",
    /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g,
  ],
];
const reservedEmail = /@(?:example\.(?:com|org|net|invalid)|[^@]+\.invalid)$/i;

/** Findings contain locations and categories only, never the sensitive value. */
export function inspectText(text) {
  return patterns.flatMap(([category, pattern]) => {
    pattern.lastIndex = 0;
    return [...text.matchAll(pattern)]
      .filter((m) => category !== "personal-email" || !reservedEmail.test(m[0]))
      .map((m) => ({
        category,
        line: text.slice(0, m.index).split("\n").length,
      }));
  });
}
export function inspectPath(path) {
  return /(?:^|\/)(?:\.env(?:\..*)?|\.netrc|\.npmrc|id_(?:rsa|dsa|ecdsa|ed25519)|[^/]+\.(?:key|p12|pfx))$|(?:^|\/)(?:\.ssh|\.aws|\.azure|\.codex|node_modules|dist|test-results|playwright-report)(?:\/|$)/i.test(
    path,
  )
    ? ["sensitive-file"]
    : [];
}
export function inspectImage(bytes, extension) {
  if (extension === ".png") {
    let at = 8;
    while (at + 12 <= bytes.length) {
      const length = bytes.readUInt32BE(at),
        kind = bytes.toString("ascii", at + 4, at + 8);
      if (["tEXt", "iTXt", "zTXt", "eXIf"].includes(kind))
        return ["image-metadata"];
      if (at + 12 + length > bytes.length) return ["invalid-image"];
      at += length + 12;
    }
  } else if ([".jpg", ".jpeg"].includes(extension)) {
    let at = 2;
    while (at + 4 <= bytes.length && bytes[at] === 255) {
      const marker = bytes[at + 1];
      at += 2;
      if (marker === 218 || marker === 217) break;
      if (marker === 216 || (marker >= 208 && marker <= 215)) continue;
      const length = bytes.readUInt16BE(at);
      if (length < 2 || at + length > bytes.length) return ["invalid-image"];
      // EXIF/XMP, IPTC and comments can identify the source device/person.
      // ICC color profiles and JFIF are display data and are preserved.
      if ([225, 237, 254].includes(marker)) return ["image-metadata"];
      at += length;
    }
  }
  return [];
}

const git = (args, options = {}) =>
  execFileSync("git", args, { maxBuffer: 32 * 1024 * 1024, ...options });
export function scan({ ref, base } = {}) {
  if (ref)
    ref = git(
      ["rev-parse", "--verify", "--end-of-options", `${ref}^{commit}`],
      { encoding: "utf8" },
    ).trim();
  const paths = git(
    ref ? ["ls-tree", "-r", "-z", "--name-only", ref] : ["ls-files", "-z"],
    { encoding: "utf8" },
  )
    .split("\0")
    .filter(Boolean);
  const findings = [];
  for (const path of paths) {
    for (const category of inspectPath(path)) findings.push({ path, category });
    if (!ref && lstatSync(path).isSymbolicLink()) {
      findings.push({ path, category: "symlink-review" });
      continue;
    }
    const bytes = ref ? git(["show", `${ref}:${path}`]) : readFileSync(path);
    const extension = extname(path).toLowerCase();
    if ([".png", ".jpg", ".jpeg"].includes(extension)) {
      for (const category of inspectImage(bytes, extension))
        findings.push({ path, category });
    } else
      for (const finding of inspectText(bytes.toString("utf8")))
        findings.push({ path, ...finding });
  }
  let commits = 0;
  if (ref && base) {
    base = git(
      ["rev-parse", "--verify", "--end-of-options", `${base}^{commit}`],
      { encoding: "utf8" },
    ).trim();
    const ids = git(["rev-list", `${base}..${ref}`], { encoding: "utf8" })
      .trim()
      .split("\n")
      .filter(Boolean);
    commits = ids.length;
    for (const id of ids) {
      const metadata = git(
        ["show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", id],
        { encoding: "utf8" },
      );
      for (const finding of inspectText(metadata))
        findings.push({ commit: id.slice(0, 12), ...finding });
    }
  }
  return { files: paths.length, commits, findings };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const args = process.argv.slice(2),
    arg = (flag) =>
      args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined;
  const report = scan({ ref: arg("--ref"), base: arg("--base") });
  console.log(JSON.stringify(report, null, 2));
  if (report.findings.length) process.exitCode = 1;
}
