import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Search,
    Clock,
    FileText,
    TrendingUp,
    TrendingDown,
    Minus,
    Eye,
    Trash2,
    Calendar,
    Loader2,
    AlertCircle,
} from 'lucide-react';
import axios from 'axios';

interface HistoryItem {
    _id: string;
    upload_id: string;
    analysis_type?: string; // 'file_upload' or 'product_url'
    metadata: {
        source_file?: string;
        file_type?: string;
        total_entries?: number;
        // Product analysis fields
        source_url?: string;
        platform?: string;
        product_name?: string;
        price?: number;
        rating?: number;
    };
    statistics: {
        total_entries: number;
        average_sentiment: number;
        sentiment_distribution: {
            positive: number;
            neutral: number;
            negative: number;
        };
        risk_score?: number;
    };
    created_at: string;
}

interface HistoryPageProps {
    onSelectAnalysis: (analysisId: string) => void;
}

const HistoryPage: React.FC<HistoryPageProps> = ({ onSelectAnalysis }) => {
    const [analyses, setAnalyses] = useState<HistoryItem[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [totalCount, setTotalCount] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    useEffect(() => {
        loadHistory();
    }, [currentPage]);

    const loadHistory = async () => {
        setIsLoading(true);
        setError(null);
        
        try {
            const skip = (currentPage - 1) * itemsPerPage;
            const response = await axios.get(`http://localhost:8000/history?skip=${skip}&limit=${itemsPerPage}`);
            setAnalyses(response.data.analyses);
            setTotalCount(response.data.total);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to load history');
        } finally {
            setIsLoading(false);
        }
    };

    const handleSearch = async () => {
        if (!searchQuery.trim()) {
            loadHistory();
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const response = await axios.get(`http://localhost:8000/history/search?q=${encodeURIComponent(searchQuery)}&skip=0&limit=${itemsPerPage}`);
            setAnalyses(response.data.analyses);
            setTotalCount(response.data.analyses.length);
            setCurrentPage(1);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Search failed');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDelete = async (analysisId: string, fileName: string) => {
        if (!window.confirm(`Delete analysis for "${fileName}"?`)) {
            return;
        }

        try {
            await axios.delete(`http://localhost:8000/history/${analysisId}`);
            loadHistory();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to delete analysis');
        }
    };

    const getSentimentColor = (score: number) => {
        if (score > 0.2) return 'text-green-600';
        if (score < -0.2) return 'text-red-600';
        return 'text-neutral-600';
    };

    const getSentimentIcon = (score: number) => {
        if (score > 0.2) return <TrendingUp className="w-4 h-4" />;
        if (score < -0.2) return <TrendingDown className="w-4 h-4" />;
        return <Minus className="w-4 h-4" />;
    };

    const totalPages = Math.ceil(totalCount / itemsPerPage);

    return (
        <div className="min-h-screen p-6 bg-neutral-50">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold text-neutral-900 mb-2 flex items-center gap-3">
                        <Clock className="w-8 h-8 text-primary-600" />
                        Analysis History
                    </h1>
                    <p className="text-neutral-600">
                        View and manage all your previous sentiment analyses
                    </p>
                </motion.div>

                {/* Search Bar */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="card mb-6"
                >
                    <div className="flex gap-3">
                        <div className="flex-1 relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" />
                            <input
                                type="text"
                                placeholder="Search by filename or content..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                                className="w-full pl-10 pr-4 py-3 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                            />
                        </div>
                        <button
                            onClick={handleSearch}
                            className="px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
                        >
                            Search
                        </button>
                        {searchQuery && (
                            <button
                                onClick={() => {
                                    setSearchQuery('');
                                    loadHistory();
                                }}
                                className="px-6 py-3 bg-neutral-200 text-neutral-700 rounded-lg font-medium hover:bg-neutral-300 transition-colors"
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </motion.div>

                {/* Loading State */}
                {isLoading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
                        <span className="ml-3 text-neutral-600">Loading history...</span>
                    </div>
                )}

                {/* Error State */}
                {error && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="card bg-red-50 border-red-200 mb-6"
                    >
                        <div className="flex items-center gap-3 text-red-700">
                            <AlertCircle className="w-5 h-5" />
                            <span>{error}</span>
                        </div>
                    </motion.div>
                )}

                {/* History List */}
                {!isLoading && !error && analyses.length === 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="card text-center py-12"
                    >
                        <FileText className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-neutral-900 mb-2">No analyses found</h3>
                        <p className="text-neutral-600">
                            {searchQuery ? 'Try a different search term' : 'Upload a file to get started'}
                        </p>
                    </motion.div>
                )}

                {!isLoading && !error && analyses.length > 0 && (
                    <div className="space-y-4">
                        {analyses.map((analysis, index) => {
                            // Check if it's a file upload or product analysis
                            const isProductAnalysis = analysis.analysis_type === 'product_url';
                            const title = isProductAnalysis 
                                ? (analysis.metadata?.product_name || 'Product Analysis')
                                : (analysis.metadata?.source_file || 'Unknown File');
                            const fileType = isProductAnalysis 
                                ? (analysis.metadata?.platform || 'Product')
                                : (analysis.metadata?.file_type || 'File');
                            
                            return (
                                <motion.div
                                    key={analysis._id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="card hover:shadow-medium transition-shadow cursor-pointer"
                                    onClick={() => onSelectAnalysis(analysis._id)}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3 mb-3">
                                                <FileText className="w-5 h-5 text-primary-600" />
                                                <h3 className="text-lg font-semibold text-neutral-900">
                                                    {title}
                                                </h3>
                                                <span className="text-xs px-2 py-1 bg-neutral-100 text-neutral-600 rounded">
                                                    {fileType?.toUpperCase()}
                                                </span>
                                            </div>

                                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                                                <div>
                                                    <div className="text-xs text-neutral-500 mb-1">Entries</div>
                                                    <div className="text-lg font-semibold text-neutral-900">
                                                        {analysis.statistics?.total_entries || 0}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-neutral-500 mb-1">Avg Sentiment</div>
                                                    <div className={`text-lg font-semibold flex items-center gap-1 ${getSentimentColor(analysis.statistics?.average_sentiment || 0)}`}>
                                                        {getSentimentIcon(analysis.statistics?.average_sentiment || 0)}
                                                        {(analysis.statistics?.average_sentiment || 0).toFixed(2)}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-neutral-500 mb-1">Risk Score</div>
                                                    <div className="text-lg font-semibold text-neutral-900">
                                                        {analysis.statistics?.risk_score || 'N/A'}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div className="text-xs text-neutral-500 mb-1">Distribution</div>
                                                    <div className="flex gap-2">
                                                        <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">
                                                            {analysis.statistics?.sentiment_distribution?.positive || 0}
                                                        </span>
                                                        <span className="text-xs px-2 py-1 bg-neutral-100 text-neutral-700 rounded">
                                                            {analysis.statistics?.sentiment_distribution?.neutral || 0}
                                                        </span>
                                                        <span className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded">
                                                            {analysis.statistics?.sentiment_distribution?.negative || 0}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="flex items-center gap-4 text-xs text-neutral-500">
                                                <div className="flex items-center gap-1">
                                                    <Calendar className="w-3 h-3" />
                                                    {new Date(analysis.created_at).toLocaleString()}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2 ml-4">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onSelectAnalysis(analysis._id);
                                                }}
                                                className="p-2 text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                                                title="View Details"
                                            >
                                                <Eye className="w-5 h-5" />
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDelete(analysis._id, title);
                                                }}
                                                className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                                                title="Delete"
                                            >
                                                <Trash2 className="w-5 h-5" />
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>
                )}                {/* Pagination */}
                {!isLoading && !error && totalPages > 1 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center justify-center gap-2 mt-8"
                    >
                        <button
                            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                            disabled={currentPage === 1}
                            className="px-4 py-2 bg-white border border-neutral-200 rounded-lg text-neutral-700 hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            Previous
                        </button>
                        
                        <div className="flex items-center gap-2">
                            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                                let pageNum;
                                if (totalPages <= 5) {
                                    pageNum = i + 1;
                                } else if (currentPage <= 3) {
                                    pageNum = i + 1;
                                } else if (currentPage >= totalPages - 2) {
                                    pageNum = totalPages - 4 + i;
                                } else {
                                    pageNum = currentPage - 2 + i;
                                }

                                return (
                                    <button
                                        key={pageNum}
                                        onClick={() => setCurrentPage(pageNum)}
                                        className={`px-4 py-2 rounded-lg transition-colors ${
                                            currentPage === pageNum
                                                ? 'bg-primary-600 text-white'
                                                : 'bg-white border border-neutral-200 text-neutral-700 hover:bg-neutral-50'
                                        }`}
                                    >
                                        {pageNum}
                                    </button>
                                );
                            })}
                        </div>

                        <button
                            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                            disabled={currentPage === totalPages}
                            className="px-4 py-2 bg-white border border-neutral-200 rounded-lg text-neutral-700 hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            Next
                        </button>
                    </motion.div>
                )}
            </div>
        </div>
    );
};

export default HistoryPage;
