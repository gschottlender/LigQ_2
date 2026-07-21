import { BrowserRouter, useLocation } from 'react-router-dom';
import { VisualizeResults } from './pages/home/VisualizeResults';
import { ConfigureSearch } from './pages/config/ConfigureSearch';
import { HelpPage } from './pages/help/HelpPage';
import { Header } from './components/Header';
import { InitialSetupGate } from './components/InitialSetupGate';
import { DatabaseProvider } from './context/DatabaseContext';

function Layout() {
  const { pathname } = useLocation();
  const isConfigure = pathname.startsWith('/configure');
  const isHelp = pathname.startsWith('/help');

  return (
    <>
      <Header />
      <div className={isConfigure ? 'block' : 'hidden'}>
        <ConfigureSearch />
      </div>
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
    <DatabaseProvider>
      <BrowserRouter>
        <InitialSetupGate>
          <Layout />
        </InitialSetupGate>
      </BrowserRouter>
    </DatabaseProvider>
  );
}
