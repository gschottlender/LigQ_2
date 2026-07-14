import { useState } from 'react';
import { Database, Layers } from 'lucide-react';
import { AddNewDatabase } from './AddNewDatabase';
import { AddNewRepresentation } from './AddNewRepresentation';

type Tab = 'database' | 'representation';

export function ConfigureSearch() {
  const [activeTab, setActiveTab] = useState<Tab>('database');

  return (
    <div className="min-h-[calc(100vh-56px)] bg-[#f8f9fa] dark:bg-[#161c23] p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-xl font-semibold font-dm-sans text-gray-800 dark:text-gray-100 mb-1">
          Manage Resources
        </h1>
        <p className="text-sm text-gray-400 dark:text-gray-500 mb-6">
          Manage compound databases and molecular representations.
        </p>

        {/* Sub-tab bar */}
        <div className="flex gap-1 bg-gray-200 dark:bg-gray-700/60 p-1 rounded-xl mb-8">
          <button
            onClick={() => setActiveTab('database')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm cursor-pointer font-medium font-dm-sans transition-all
              ${activeTab === 'database'
                ? 'bg-white dark:bg-gray-600 text-gray-800 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }`}
          >
            <Database className="w-4 h-4" />
            Add new database
          </button>
          <button
            onClick={() => setActiveTab('representation')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm cursor-pointer font-medium font-dm-sans transition-all
              ${activeTab === 'representation'
                ? 'bg-white dark:bg-gray-600 text-gray-800 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
              }`}
          >
            <Layers className="w-4 h-4" />
            Add new representation
          </button>
        </div>

        {/* Content */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700/60 p-6 shadow-sm">
          {activeTab === 'database' ? <AddNewDatabase /> : <AddNewRepresentation />}
        </div>
      </div>
    </div>
  );
}
