import { describe, expect, test } from "vitest";
import {
  inspectText,
  inspectImage,
  inspectPath,
} from "../../scripts/privacy-check.mjs";

describe("publication privacy scan", () => {
  test("detects secrets and personal locations without returning the matched value", () => {
    const secret = "ghp_" + "x".repeat(36);
    const path = ["", "Users", "example", "private.txt"].join("/");
    const network = [192, 168, 0, 7].join(".");
    const findings = inspectText([secret, path, network].join("\n"));
    expect(findings.map((f) => f.category)).toEqual(
      expect.arrayContaining([
        "credential-token",
        "personal-path",
        "private-network",
      ]),
    );
    expect(JSON.stringify(findings)).not.toContain(secret);
  });
  test("recognizes generic local preview URLs and public revision hashes as non-sensitive", () => {
    expect(
      inspectText(
        "http://127.0.0.1:4173 abcdef0123456789abcdef0123456789abcdef0123",
      ),
    ).toEqual([]);
  });
  test("detects non-anonymous emails and credential-bearing URLs", () => {
    const email = ["person", "company.internal"].join("@");
    const url = "https://" + ["user", "password"].join(":") + "@host.test";
    expect(inspectText(email + "\n" + url).map((f) => f.category)).toEqual(
      expect.arrayContaining(["personal-email", "authenticated-url"]),
    );
    expect(inspectText(["project", "example.invalid"].join("@"))).toEqual([]);
  });
  test("rejects credential files and private-key payloads", () => {
    expect(inspectPath(".env.production")).toContain("sensitive-file");
    expect(inspectPath("config/id_rsa")).toContain("sensitive-file");
    const key = ["-----BEGIN", "OPENSSH", "PRIVATE KEY-----"].join(" ");
    expect(inspectText(key).map((f) => f.category)).toContain("private-key");
    expect(inspectPath("src/content/models.ts")).toEqual([]);
  });
  test("detects PNG text and JPEG EXIF metadata without flagging a color profile", () => {
    const header = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    const chunk = Buffer.alloc(12);
    chunk.write("tEXt", 4);
    expect(inspectImage(Buffer.concat([header, chunk]), ".png")).toContain(
      "image-metadata",
    );
    const jpeg = (payload: Buffer) => {
      const length = Buffer.alloc(2);
      length.writeUInt16BE(payload.length + 2);
      return Buffer.concat([
        Buffer.from([255, 216, 255, 225]),
        length,
        payload,
        Buffer.from([255, 217]),
      ]);
    };
    expect(inspectImage(jpeg(Buffer.from("Exif\0\0")), ".jpg")).toContain(
      "image-metadata",
    );
    const color = jpeg(Buffer.from("ICC_PROFILE\0"));
    color[3] = 226;
    expect(inspectImage(color, ".jpg")).toEqual([]);
  });
});

test("quoted JSON credentials and encrypted private keys are detected", () => {
  const payload = JSON.stringify({
    ["api" + "_key"]: ["secret", "value"].join("-"),
  });
  expect(inspectText(payload).map((f) => f.category)).toContain(
    "credential-assignment",
  );
  const header = ["-----BEGIN", "ENCRYPTED", "PRIVATE KEY-----"].join(" ");
  expect(inspectText(header).map((f) => f.category)).toContain("private-key");
});
