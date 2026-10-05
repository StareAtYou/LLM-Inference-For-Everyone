import { useState } from "react";
import { Link as LinkIcon } from "lucide-react";
export default function ShareConfig({
  label = "分享配置",
}: {
  label?: string;
}) {
  const [fallback, setFallback] = useState(false);
  const [copied, setCopied] = useState(false);
  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
    } catch {
      setFallback(true);
    }
  }
  return (
    <div className="share-config">
      <button className="control-button" onClick={share}>
        <LinkIcon size={15} />
        {copied ? "链接已复制" : label}
      </button>
      {fallback && (
        <label className="small-note">
          复制此链接
          <input
            aria-label="配置分享链接"
            readOnly
            value={window.location.href}
            onFocus={(e) => e.target.select()}
          />
        </label>
      )}
    </div>
  );
}
