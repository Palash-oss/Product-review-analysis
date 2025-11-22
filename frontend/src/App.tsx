import { useState } from 'react';
import UploadPage from './pages/UploadPage';
import Dashboard from './pages/Dashboard';
import HistoryPage from './pages/HistoryPage';
import ProductAnalysisPage from './pages/ProductAnalysisPage';
import type { ProcessedDataset } from './types';
import './index.css';

type View = 'upload' | 'dashboard' | 'history' | 'product-analysis';

function App() {
  const [currentDataset, setCurrentDataset] = useState<ProcessedDataset | null>(null);
  const [currentView, setCurrentView] = useState<View>('upload');
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);

  const handleUploadComplete = (dataset: ProcessedDataset) => {
    setCurrentDataset(dataset);
    setCurrentView('dashboard');
  };

  const handleBack = () => {
    setCurrentDataset(null);
    setSelectedAnalysisId(null);
    setCurrentView('upload');
  };

  const handleViewHistory = () => {
    setCurrentView('history');
  };

  const handleSelectAnalysis = (analysisId: string) => {
    setSelectedAnalysisId(analysisId);
    // TODO: Load analysis data from backend and show in dashboard
    alert(`Analysis ID: ${analysisId}\nFull history detail view coming soon!`);
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Navigation Header */}
      <div className="bg-white border-b border-neutral-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-primary-600">Sentiment Analysis Platform</h1>
          <div className="flex gap-3">
            <button
              onClick={() => setCurrentView('product-analysis')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                currentView === 'product-analysis'
                  ? 'bg-primary-600 text-white'
                  : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
              }`}
            >
              Product Analysis
            </button>
            <button
              onClick={() => setCurrentView('upload')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                currentView === 'upload'
                  ? 'bg-primary-600 text-white'
                  : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
              }`}
            >
              Upload File
            </button>
            <button
              onClick={handleViewHistory}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                currentView === 'history'
                  ? 'bg-primary-600 text-white'
                  : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'
              }`}
            >
              History
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {currentView === 'product-analysis' && (
        <ProductAnalysisPage />
      )}
      {currentView === 'upload' && (
        <UploadPage onUploadComplete={handleUploadComplete} />
      )}
      {currentView === 'dashboard' && currentDataset && (
        <Dashboard dataset={currentDataset} onBack={handleBack} />
      )}
      {currentView === 'history' && (
        <HistoryPage onSelectAnalysis={handleSelectAnalysis} />
      )}
    </div>
  );
}

export default App;
