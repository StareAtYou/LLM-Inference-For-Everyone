import { lazy, Suspense, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router";
import { PreferencesProvider } from "./hooks/usePreferences";
import SiteShell from "./components/SiteShell";
const Lab = lazy(() => import("./pages/Lab"));
const Frameworks = lazy(() => import("./pages/Frameworks"));
const Models = lazy(() => import("./pages/Models"));
const Pipeline = lazy(() => import("./pages/Pipeline"));
const Learn = lazy(() => import("./pages/Learn"));
const Topic = lazy(() => import("./pages/Topic"));
const Home = lazy(() => import("./pages/Home"));
const NotFound = lazy(() => import("./pages/NotFound"));
export default function App() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return (
    <PreferencesProvider>
      <SiteShell>
        <Suspense
          fallback={
            <div className="page-loading" role="status">
              正在打开图谱…
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/lab/:experiment" element={<Lab />} />
            <Route path="/lab" element={<Lab />} />
            <Route path="/frameworks/:framework" element={<Frameworks />} />
            <Route path="/models" element={<Models />} />
            <Route path="/pipeline" element={<Pipeline />} />
            <Route path="/learn" element={<Learn />} />
            <Route path="/learn/:topic" element={<Topic />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </SiteShell>
    </PreferencesProvider>
  );
}
