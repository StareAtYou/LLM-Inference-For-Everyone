import { useEffect, useState, useRef, type ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router";
import { Menu, X, Code2 } from "lucide-react";
export function BrandMark() {
  return (
    <svg
      width="30"
      height="30"
      viewBox="0 0 30 30"
      fill="none"
      aria-hidden="true"
    >
      <path d="M15 2 28 15 15 28 2 15Z" stroke="currentColor" strokeWidth="2" />
      <path d="m15 8 7 7-7 7-7-7Z" fill="currentColor" />
      <path
        d="M15 0v7m15 8h-7M15 30v-7M0 15h7"
        stroke="currentColor"
        strokeWidth="2"
      />
    </svg>
  );
}
const navigation = [
  ["学习地图", "/learn"],
  ["推理流程", "/pipeline"],
  ["多 GPU 与通信", "/distributed"],
  ["模型结构", "/models"],
  ["优化实验室", "/lab/kv-cache"],
  ["框架源码", "/frameworks/vllm"],
];
export default function SiteShell({ children }: { children: ReactNode }) {
  const menuButton = useRef<HTMLButtonElement>(null);
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => {
    setOpen(false);
  }, [pathname]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        if (document.activeElement?.closest(".main-nav"))
          menuButton.current?.focus();
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        navigate("/learn?focus=search");
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [navigate]);
  return (
    <>
      <a className="skip-link" href="#main">
        跳到内容
      </a>
      <header className="site-header">
        <div className="header-inner">
          <Link to="/" className="brand" aria-label="推理图谱首页">
            <BrandMark />
            <span>
              推理图谱<small>INFERENCE ATLAS</small>
            </span>
          </Link>
          <nav
            className={open ? "main-nav is-open" : "main-nav"}
            aria-label="主要导航"
          >
            {navigation.map(([name, to]) => (
              <NavLink
                key={to}
                to={to}
                className={
                  pathname.startsWith(
                    to.startsWith("/lab")
                      ? "/lab"
                      : to.startsWith("/frameworks")
                        ? "/frameworks"
                        : to,
                  )
                    ? "active"
                    : ""
                }
              >
                {name}
              </NavLink>
            ))}
          </nav>
          <div className="header-actions">
            <Link to="/learn" className="index-link">
              探索图谱 <span className="keycap">⌘ K</span>
            </Link>
            <button
              ref={menuButton}
              className="icon-button mobile-menu"
              aria-label="展开导航"
              aria-expanded={open}
              onClick={() => setOpen(!open)}
            >
              {open ? <X size={22} /> : <Menu size={22} />}
            </button>
          </div>
        </div>
      </header>
      <main id="main">{children}</main>
      <footer className="site-footer">
        <div className="footer-inner">
          <Link to="/" className="brand">
            <BrandMark />
            <span>
              推理图谱<small>INFERENCE ATLAS</small>
            </span>
          </Link>
          <p>从计算原理，到工程实现。</p>
          <a
            href="https://github.com/vllm-project/vllm"
            target="_blank"
            rel="noreferrer"
          >
            <Code2 size={16} /> 开源，从这里继续
          </a>
          <span className="mono">VOL. 01 / 2026</span>
        </div>
      </footer>
    </>
  );
}
