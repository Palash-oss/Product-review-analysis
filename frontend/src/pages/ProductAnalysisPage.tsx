import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Link2,
    Search,
    ShoppingCart,
    TrendingUp,
    AlertCircle,
    CheckCircle,
    XCircle,
    Loader2,
    Star,
    ThumbsUp,
    ThumbsDown,
    DollarSign,
    Package,
    Award,
    Info,
} from 'lucide-react';
import axios from 'axios';
import {
    PieChart,
    Pie,
    Cell,
    Tooltip,
    ResponsiveContainer,
} from 'recharts';interface ProductAnalysisResult {
    success: boolean;
    product: {
        name: string;
        platform: string;
        price: number;
        currency: string;
        rating: number;
        image_url: string;
        url: string;
    };
    sentiment_analysis: {
        total_reviews: number;
        average_sentiment: number;
        distribution: {
            positive: number;
            neutral: number;
            negative: number;
        };
    };
    recommendation: {
        score: number;
        verdict: string;
        confidence: string;
        reasoning: string;
        should_buy: boolean;
    };
    detailed_analysis: any;
    pros: string[];
    cons: string[];
    summary: string;
    alternatives?: Array<{
        name: string;
        url: string;
        price: number;
        rating: number;
        review_count: number;
        image_url: string;
        platform: string;
    }>;
}

const ProductAnalysisPage: React.FC = () => {
    const [productUrl, setProductUrl] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [analysis, setAnalysis] = useState<ProductAnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleAnalyze = async () => {
        if (!productUrl.trim()) {
            setError('Please enter a product URL');
            return;
        }

        // Basic URL validation
        try {
            new URL(productUrl);
        } catch {
            setError('Please enter a valid URL');
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysis(null);

        try {
            const response = await axios.post('http://localhost:8000/analyze-product', {
                url: productUrl,
            });

            if (response.data.success === false) {
                // Handle case where scraping succeeded but no reviews found
                setError(response.data.message || 'No reviews found for this product');
                return;
            }

            setAnalysis(response.data);
        } catch (err: any) {
            console.error('Analysis error:', err);
            const errorDetail = err.response?.data?.detail || err.response?.data?.message || err.message;
            setError(errorDetail || 'Failed to analyze product. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const getVerdictColor = (verdict: string) => {
        if (verdict.includes('NOT RECOMMENDED')) return 'text-red-600';
        if (verdict.includes('STRONGLY RECOMMEND')) return 'text-green-600';
        if (verdict.includes('RECOMMEND')) return 'text-green-500';
        if (verdict.includes('CAUTIOUSLY')) return 'text-yellow-600';
        if (verdict.includes('NEUTRAL')) return 'text-neutral-600';
        return 'text-neutral-700';
    };

    const getVerdictIcon = (verdict: string) => {
        if (verdict.includes('NOT RECOMMENDED')) return <XCircle className="w-12 h-12" />;
        if (verdict.includes('STRONGLY RECOMMEND')) return <CheckCircle className="w-12 h-12" />;
        if (verdict.includes('RECOMMEND')) return <ThumbsUp className="w-12 h-12" />;
        if (verdict.includes('CAUTIOUSLY')) return <AlertCircle className="w-12 h-12" />;
        if (verdict.includes('NEUTRAL')) return <Info className="w-12 h-12" />;
        return <Info className="w-12 h-12" />;
    };

    const getVerdictBgColor = (verdict: string) => {
        if (verdict.includes('NOT RECOMMENDED')) return 'bg-red-50 border-red-200';
        if (verdict.includes('STRONGLY RECOMMEND')) return 'bg-green-50 border-green-200';
        if (verdict.includes('RECOMMEND')) return 'bg-green-50 border-green-200';
        if (verdict.includes('CAUTIOUSLY')) return 'bg-yellow-50 border-yellow-200';
        if (verdict.includes('NEUTRAL')) return 'bg-neutral-50 border-neutral-200';
        return 'bg-neutral-50 border-neutral-200';
    };

    const renderSentimentChart = () => {
        if (!analysis?.sentiment_analysis?.distribution) return null;

        const data = [
            {
                name: 'Positive',
                value: analysis.sentiment_analysis.distribution.positive || 0,
                color: '#22c55e',
            },
            {
                name: 'Neutral',
                value: analysis.sentiment_analysis.distribution.neutral || 0,
                color: '#64748b',
            },
            {
                name: 'Negative',
                value: analysis.sentiment_analysis.distribution.negative || 0,
                color: '#ef4444',
            },
        ];

        return (
            <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                    </Pie>
                    <Tooltip />
                </PieChart>
            </ResponsiveContainer>
        );
    };

    const renderScoreBar = () => {
        if (!analysis?.recommendation?.score) return null;

        const score = analysis.recommendation.score;
        const color =
            score >= 75
                ? 'bg-green-500'
                : score >= 55
                ? 'bg-yellow-500'
                : 'bg-red-500';

        return (
            <div className="w-full bg-neutral-200 rounded-full h-4 overflow-hidden">
                <div
                    className={`h-full ${color} transition-all duration-1000 ease-out`}
                    style={{ width: `${score}%` }}
                />
            </div>
        );
    };

    return (
        <div className="min-h-screen p-6 bg-neutral-50">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold text-neutral-900 mb-2 flex items-center gap-3">
                        <ShoppingCart className="w-8 h-8 text-primary-600" />
                        AI Product Analysis
                    </h1>
                    <p className="text-neutral-600">
                        Get AI-powered insights before you buy. Paste any Amazon, Flipkart, or Myntra product link.
                    </p>
                </motion.div>

                {/* URL Input */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="card mb-6"
                >
                    <div className="flex gap-3">
                        <div className="flex-1 relative">
                            <Link2 className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" />
                            <input
                                type="text"
                                placeholder="https://www.amazon.in/product/..."
                                value={productUrl}
                                onChange={(e) => setProductUrl(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                                className="w-full pl-10 pr-4 py-3 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                                disabled={isLoading}
                            />
                        </div>
                        <button
                            onClick={handleAnalyze}
                            disabled={isLoading}
                            className="px-8 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Analyzing...
                                </>
                            ) : (
                                <>
                                    <Search className="w-5 h-5" />
                                    Analyze
                                </>
                            )}
                        </button>
                    </div>

                    {/* Supported platforms */}
                    <div className="mt-4 text-sm text-neutral-500">
                        <span className="font-medium">Supported:</span> Amazon, Flipkart, Myntra, and more
                    </div>
                </motion.div>

                {/* Error Message */}
                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="card bg-red-50 border-red-200 mb-6"
                        >
                            <div className="flex items-center gap-3 text-red-700">
                                <AlertCircle className="w-5 h-5" />
                                <span>{error}</span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Loading State */}
                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="card text-center py-12"
                    >
                        <Loader2 className="w-16 h-16 text-primary-600 animate-spin mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-neutral-900 mb-2">Analyzing Product...</h3>
                        <p className="text-neutral-600">
                            Scraping reviews, analyzing sentiment, and generating AI insights...
                        </p>
                        
                        {/* Animated progress bar */}
                        <div className="mt-6 max-w-md mx-auto">
                            <div className="w-full h-2 bg-neutral-200 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-gradient-to-r from-green-500 via-green-600 to-green-700"
                                    initial={{ width: '0%' }}
                                    animate={{ width: '100%' }}
                                    transition={{ duration: 8, ease: 'linear' }}
                                />
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* Analysis Results */}
                <AnimatePresence>
                    {analysis && !isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="space-y-6"
                        >
                            {/* Product Header */}
                            <div className="card">
                                <div className="flex items-start gap-6">
                                    {analysis?.product?.image_url && (
                                        <img
                                            src={analysis.product.image_url}
                                            alt={analysis.product.name}
                                            className="w-32 h-32 object-contain rounded-lg border border-neutral-200"
                                        />
                                    )}
                                    <div className="flex-1">
                                        <h2 className="text-2xl font-bold text-neutral-900 mb-2">
                                            {analysis?.product?.name || 'Unknown Product'}
                                        </h2>
                                        <div className="flex items-center gap-4 text-neutral-600 mb-4">
                                            <span className="flex items-center gap-1">
                                                <Package className="w-4 h-4" />
                                                {analysis?.product?.platform || 'Unknown'}
                                            </span>
                                            {analysis?.product?.price && (
                                                <span className="flex items-center gap-1 font-semibold text-lg text-primary-600">
                                                    <DollarSign className="w-4 h-4" />
                                                    {analysis.product.currency} {analysis.product.price}
                                                </span>
                                            )}
                                            {analysis?.product?.rating && (
                                                <span className="flex items-center gap-1">
                                                    <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                                                    {analysis.product.rating}/5
                                                </span>
                                            )}
                                        </div>
                                        {analysis?.product?.url && (
                                            <a
                                                href={analysis.product.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-primary-600 hover:underline text-sm"
                                            >
                                                View on {analysis.product.platform} →
                                            </a>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* AI Recommendation */}
                            <div className={`card ${getVerdictBgColor(analysis?.recommendation?.verdict || '')}`}>
                                <div className="flex items-start gap-6">
                                    <div className={`${getVerdictColor(analysis?.recommendation?.verdict || '')}`}>
                                        {getVerdictIcon(analysis?.recommendation?.verdict || '')}
                                    </div>
                                    <div className="flex-1">
                                        <h3 className={`text-2xl font-bold mb-2 ${getVerdictColor(analysis?.recommendation?.verdict || '')}`}>
                                            {analysis?.recommendation?.verdict || 'Analyzing...'}
                                        </h3>
                                        <p className="text-neutral-700 mb-4">{analysis?.recommendation?.reasoning || ''}</p>
                                        
                                        <div className="mb-4">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-sm font-medium text-neutral-700">
                                                    Recommendation Score
                                                </span>
                                                <span className="text-2xl font-bold text-neutral-900">
                                                    {analysis?.recommendation?.score || 0}/100
                                                </span>
                                            </div>
                                            {renderScoreBar()}
                                        </div>
                                        
                                        <div className="flex items-center gap-2 text-sm">
                                            <Award className="w-4 h-4 text-neutral-600" />
                                            <span className="text-neutral-600">
                                                Confidence: <span className="font-semibold">{analysis?.recommendation?.confidence || 'Unknown'}</span>
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Sentiment Analysis */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="card">
                                    <h3 className="text-lg font-semibold text-neutral-900 mb-4 flex items-center gap-2">
                                        <TrendingUp className="w-5 h-5 text-primary-600" />
                                        Sentiment Distribution
                                    </h3>
                                    {renderSentimentChart()}
                                    <div className="mt-4 text-center text-neutral-600">
                                        <span className="text-2xl font-bold text-neutral-900">
                                            {analysis?.sentiment_analysis?.total_reviews || 0}
                                        </span>
                                        {' '}reviews analyzed
                                    </div>
                                </div>

                                {/* Quick Stats */}
                                <div className="card">
                                    <h3 className="text-lg font-semibold text-neutral-900 mb-4">Quick Stats</h3>
                                    <div className="space-y-4">
                                        <div>
                                            <div className="text-sm text-neutral-600 mb-1">Average Sentiment</div>
                                            <div className="text-2xl font-bold text-neutral-900">
                                                {analysis?.sentiment_analysis?.average_sentiment?.toFixed(2) || '0.00'}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-neutral-600 mb-1">Sentiment Grade</div>
                                            <div className="text-2xl font-bold text-neutral-900">
                                                {analysis?.detailed_analysis?.sentiment_breakdown?.grade || 'N/A'}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="text-sm text-neutral-600 mb-1">Risk Level</div>
                                            <div className="text-2xl font-bold text-neutral-900">
                                                {analysis?.detailed_analysis?.risk_factors?.level || 'Unknown'}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Pros & Cons */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Pros */}
                                <div className="card bg-green-50 border-green-200">
                                    <h3 className="text-lg font-semibold text-green-900 mb-4 flex items-center gap-2">
                                        <ThumbsUp className="w-5 h-5" />
                                        What Customers Love
                                    </h3>
                                    {analysis?.pros && analysis.pros.length > 0 ? (
                                        <ul className="space-y-2">
                                            {analysis.pros.map((pro, index) => (
                                                <li key={index} className="flex items-start gap-2 text-green-800">
                                                    <CheckCircle className="w-4 h-4 mt-1 flex-shrink-0" />
                                                    <span>{pro}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="text-green-700">No specific positive highlights found</p>
                                    )}
                                </div>

                                {/* Cons */}
                                <div className="card bg-red-50 border-red-200">
                                    <h3 className="text-lg font-semibold text-red-900 mb-4 flex items-center gap-2">
                                        <ThumbsDown className="w-5 h-5" />
                                        Common Complaints
                                    </h3>
                                    {analysis?.cons && analysis.cons.length > 0 ? (
                                        <ul className="space-y-2">
                                            {analysis.cons.map((con, index) => (
                                                <li key={index} className="flex items-start gap-2 text-red-800">
                                                    <XCircle className="w-4 h-4 mt-1 flex-shrink-0" />
                                                    <span>{con}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="text-red-700">No specific complaints found</p>
                                    )}
                                </div>
                            </div>

                            {/* Summary */}
                            <div className="card">
                                <h3 className="text-lg font-semibold text-neutral-900 mb-4">AI Summary</h3>
                                <div className="prose max-w-none text-neutral-700 whitespace-pre-line">
                                    {analysis?.summary}
                                </div>
                            </div>

                            {/* Alternative Products */}
                            {analysis?.alternatives && analysis.alternatives.length > 0 && (
                                <div className="card bg-blue-50 border-blue-200">
                                    <h3 className="text-xl font-bold text-blue-900 mb-4 flex items-center gap-2">
                                        <TrendingUp className="w-6 h-6" />
                                        Better Alternatives
                                    </h3>
                                    <p className="text-blue-800 mb-6">
                                        Based on customer reviews, here are some better-rated alternatives:
                                    </p>
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {analysis.alternatives.map((alt, index) => (
                                            <motion.a
                                                key={index}
                                                href={alt.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="bg-white rounded-lg border-2 border-blue-200 p-4 hover:border-blue-400 hover:shadow-lg transition-all cursor-pointer"
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                            >
                                                {alt.image_url && (
                                                    <img
                                                        src={alt.image_url}
                                                        alt={alt.name}
                                                        className="w-full h-32 object-contain mb-3"
                                                    />
                                                )}
                                                <h4 className="font-semibold text-sm text-neutral-900 mb-2 line-clamp-2">
                                                    {alt.name}
                                                </h4>
                                                <div className="flex items-center justify-between text-sm mb-2">
                                                    {alt.price && (
                                                        <span className="font-bold text-green-700">
                                                            ₹{alt.price.toLocaleString()}
                                                        </span>
                                                    )}
                                                    {alt.rating && (
                                                        <div className="flex items-center gap-1">
                                                            <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                                                            <span className="font-medium">{alt.rating}</span>
                                                        </div>
                                                    )}
                                                </div>
                                                {alt.review_count && (
                                                    <p className="text-xs text-neutral-600">
                                                        {alt.review_count.toLocaleString()} reviews
                                                    </p>
                                                )}
                                                <div className="mt-3 flex items-center justify-between">
                                                    <span className="text-xs text-neutral-500 uppercase">
                                                        {alt.platform}
                                                    </span>
                                                    <span className="text-xs text-blue-600 font-medium">
                                                        View Product →
                                                    </span>
                                                </div>
                                            </motion.a>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default ProductAnalysisPage;
