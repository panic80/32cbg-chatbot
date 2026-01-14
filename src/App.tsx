import { Suspense, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'sonner';
import './index.css';
import { useTheme } from '@/context/ThemeContext';
import ScrollToTop from '@/components/ScrollToTop';
import RouteSkeleton from '@/components/RouteSkeleton';
import ErrorBoundary from '@/components/ErrorBoundary';
import useRoutePrefetch from '@/hooks/useRoutePrefetch';
import useMobileFlag from '@/hooks/useMobileFlag';
import useVisitAnalytics from '@/hooks/useVisitAnalytics';
import { appRoutes, isLazyRoute, type RouteContext } from '@/routes/config';

const VisitAnalyticsListener = () => {
  useVisitAnalytics();
  return null;
};

function App() {
  const { theme, toggleTheme } = useTheme();

  const routeContext = useMemo<RouteContext>(
    () => ({
      theme,
      toggleTheme,
    }),
    [theme, toggleTheme],
  );

  const prefetchTargets = useMemo(
    () =>
      appRoutes
        .filter((route) => isLazyRoute(route) && route.prefetch)
        .map((route) => (isLazyRoute(route) ? route.loader : null))
        .filter((loader): loader is NonNullable<typeof loader> => loader !== null),
    [],
  );

  useRoutePrefetch(prefetchTargets);
  useMobileFlag();

  return (
    <ErrorBoundary>
      <Router>
        <VisitAnalyticsListener />
        <ScrollToTop />
        <div
          id="app-scroll-root"
          className="w-screen min-h-screen overflow-x-hidden overflow-y-auto m-0 p-0 max-w-[100vw]"
        >
          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                background: theme === 'dark' ? '#1f2937' : '#ffffff',
                color: theme === 'dark' ? '#f3f4f6' : '#111827',
                border: theme === 'dark' ? '1px solid #374151' : '1px solid #e5e7eb',
              },
            }}
          />
          <Suspense fallback={<RouteSkeleton />}>
            <Routes>
              {appRoutes.map((route) => {
                if (!isLazyRoute(route)) {
                  return <Route key={route.path} path={route.path} element={route.element} />;
                }

                const Component = route.component;
                const componentProps: Record<string, unknown> = route.getProps
                  ? route.getProps(routeContext)
                  : {};

                return (
                  <Route
                    key={route.path}
                    path={route.path}
                    element={<Component {...componentProps} />}
                  />
                );
              })}
            </Routes>
          </Suspense>
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
