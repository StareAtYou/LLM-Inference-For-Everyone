import { Link } from "react-router";
export default function NotFound() {
  return (
    <div className="page-width empty-page">
      <span className="eyebrow">OUTSIDE THE MAP / 404</span>
      <h1>这条路径尚未收录。</h1>
      <p>可以回到图谱，寻找另一个探索入口。</p>
      <Link to="/" className="button primary">
        返回首页
      </Link>
    </div>
  );
}
