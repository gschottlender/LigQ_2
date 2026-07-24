import { BrowserRouter, Navigate, useLocation } from 'react-router-dom';
import { VisualizeResults } from './pages/home/VisualizeResults';
import { ConfigureSearch } from './pages/config/ConfigureSearch';
import { HelpPage } from './pages/help/HelpPage';
import { Header } from './components/Header';
import { InitialSetupGate } from './components/InitialSetupGate';
import { DatabaseProvider } from './context/DatabaseContext';
import { SystemPolicyProvider, useSystemPolicy } from './context/SystemPolicyContext';

function Layout() {
  const { pathname } = useLocation();
  const { isWeb } = useSystemPolicy();
  const isConfigure = !isWeb && pathname.startsWith('/configure');
  const isHelp = pathname.startsWith('/help');

  if (isWeb && pathname.startsWith('/configure')) {
    return <Navigate to="/" replace />;
  }

  return (
    <>
      <Header />
      {!isWeb && (
        <div className={isConfigure ? 'block' : 'hidden'}>
          <ConfigureSearch />
        </div>
      )}
      <div className={isHelp ? 'block' : 'hidden'}>
        <HelpPage />
      </div>
      <div className={!isConfigure && !isHelp ? 'block' : 'hidden'}>
        <VisualizeResults />
      </div>
    </>
  );
}

export function App() {
  return (
    <SystemPolicyProvider>
      <DatabaseProvider>
        <BrowserRouter>
          <InitialSetupGate>
            <Layout />
          </InitialSetupGate>
        </BrowserRouter>
      </DatabaseProvider>
    </SystemPolicyProvider>
  );
}
