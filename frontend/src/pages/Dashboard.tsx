import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    TrendingUp,
    TrendingDown,
    Minus,
    AlertCircle,
    FileText,
    Calendar,
    BarChart3,
    Eye,
    ArrowLeft,
    Lightbulb,
    Activity,
    Target,
    AlertTriangle,
    CheckCircle,
    Info,
} from 'lucide-react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
} from 'recharts';
import type { ProcessedDataset, SentimentEntry } from '../types';

interface DashboardProps {
    dataset: ProcessedDataset;
    onBack: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ dataset, onBack }) => {
    const [selectedEntry, setSelectedEntry] = useState<SentimentEntry | null>(null);
    const [activeTab, setActiveTab] = useState<'overview' | 'analytics'>('overview');

    const getSentimentColor = (label: string) => {
        switch (label) {
            case 'Positive':
                return 'text-green-600';
            case 'Negative':
                return 'text-red-600';
            default:
                return 'text-neutral-600';
        }
    };

    const getSentimentBadge = (label: string) => {
        switch (label) {
            case 'Positive':
                return 'badge-positive';
            case 'Negative':
                return 'badge-negative';
            default:
                return 'badge-neutral';
        }
    };

    const getSentimentIcon = (label: string) => {
        switch (label) {
            case 'Positive':
                return <TrendingUp className="w-4 h-4" />;
            case 'Negative':
                return <TrendingDown className="w-4 h-4" />;
            default:
                return <Minus className="w-4 h-4" />;
        }
    };

    // Prepare chart data
    const chartData = dataset.data.map((entry, index) => ({
        id: entry.id,
        index: index + 1,
        score: entry.sentiment_score,
        label: entry.sentiment_label,
        confidence: entry.confidence,
    }));

    const pieData = [
        {
            name: 'Positive',
            value: dataset.statistics.sentiment_distribution.positive || 0,
            color: '#16a34a',
        },
        {
            name: 'Neutral',
            value: dataset.statistics.sentiment_distribution.neutral || 0,
            color: '#737373',
        },
        {
            name: 'Negative',
            value: dataset.statistics.sentiment_distribution.negative || 0,
            color: '#dc2626',
        },
    ].filter(item => item.value > 0); // Only show segments with data

    const getRiskLevel = (score: number) => {
        if (score >= 70) return { label: 'High', color: 'text-red-600', bg: 'bg-red-100' };
        if (score >= 40) return { label: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-100' };
        return { label: 'Low', color: 'text-green-600', bg: 'bg-green-100' };
    };

    const riskLevel = getRiskLevel(dataset.statistics.risk_score);

    return (
        <div className="min-h-screen p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <button
                        onClick={onBack}
                        className="flex items-center gap-2 text-neutral-600 hover:text-neutral-900 mb-4 transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Upload New File</span>
                    </button>

                    <div className="flex items-start justify-between">
                        <div>
                            <h1 className="text-3xl font-bold text-neutral-900 mb-2">
                                Sentiment Analysis Dashboard
                            </h1>
                            <div className="flex items-center gap-4 text-sm text-neutral-600">
                                <div className="flex items-center gap-2">
                                    <FileText className="w-4 h-4" />
                                    <span>{dataset.metadata.source_file}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Calendar className="w-4 h-4" />
                                    <span>{new Date(dataset.processed_at).toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Tab Navigation */}
                    <div className="flex gap-2 mt-6 border-b border-neutral-200">
                        <button
                            onClick={() => setActiveTab('overview')}
                            className={`px-6 py-3 font-medium transition-colors relative ${
                                activeTab === 'overview'
                                    ? 'text-primary-600'
                                    : 'text-neutral-600 hover:text-neutral-900'
                            }`}
                        >
                            Overview
                            {activeTab === 'overview' && (
                                <motion.div
                                    layoutId="activeTab"
                                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600"
                                />
                            )}
                        </button>
                        <button
                            onClick={() => setActiveTab('analytics')}
                            className={`px-6 py-3 font-medium transition-colors relative flex items-center gap-2 ${
                                activeTab === 'analytics'
                                    ? 'text-primary-600'
                                    : 'text-neutral-600 hover:text-neutral-900'
                            }`}
                        >
                            <Activity className="w-4 h-4" />
                            AI Analytics
                            {activeTab === 'analytics' && (
                                <motion.div
                                    layoutId="activeTab"
                                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600"
                                />
                            )}
                        </button>
                    </div>
                </motion.div>

                {/* Overview Tab */}
                {activeTab === 'overview' && (
                    <>
                        {/* Statistics Cards */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
                >
                    <div className="stat-card">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium text-neutral-600">Total Entries</span>
                            <BarChart3 className="w-5 h-5 text-primary-600" />
                        </div>
                        <div className="text-3xl font-bold text-neutral-900">
                            {dataset.statistics.total_entries}
                        </div>
                    </div>

                    <div className="stat-card">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium text-neutral-600">Avg Sentiment</span>
                            <TrendingUp className="w-5 h-5 text-green-600" />
                        </div>
                        <div className="text-3xl font-bold text-neutral-900">
                            {dataset.statistics.average_sentiment.toFixed(2)}
                        </div>
                        <div className="text-xs text-neutral-500 mt-1">Scale: -1 to +1</div>
                    </div>

                    <div className="stat-card">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium text-neutral-600">Avg Confidence</span>
                            <Eye className="w-5 h-5 text-primary-600" />
                        </div>
                        <div className="text-3xl font-bold text-neutral-900">
                            {(dataset.statistics.confidence_avg * 100).toFixed(1)}%
                        </div>
                    </div>

                    <div className="stat-card">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium text-neutral-600">Risk Score</span>
                            <AlertCircle className={`w-5 h-5 ${riskLevel.color}`} />
                        </div>
                        <div className="flex items-baseline gap-2">
                            <div className="text-3xl font-bold text-neutral-900">
                                {dataset.statistics.risk_score}
                            </div>
                            <span className={`badge ${riskLevel.bg} ${riskLevel.color} border-0`}>
                                {riskLevel.label}
                            </span>
                        </div>
                    </div>
                </motion.div>

                {/* Charts Section */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    {/* Sentiment Waveform */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="lg:col-span-2 card"
                    >
                        <h2 className="text-lg font-semibold text-neutral-900 mb-6">
                            Sentiment Waveform
                        </h2>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                <XAxis
                                    dataKey="index"
                                    stroke="#737373"
                                    style={{ fontSize: '12px' }}
                                    label={{ value: 'Entry Index', position: 'insideBottom', offset: -5 }}
                                />
                                <YAxis
                                    stroke="#737373"
                                    style={{ fontSize: '12px' }}
                                    domain={[-1, 1]}
                                    label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'white',
                                        border: '1px solid #e5e5e5',
                                        borderRadius: '8px',
                                        padding: '12px',
                                    }}
                                    formatter={(value: any, name: string) => {
                                        if (name === 'score') return [value.toFixed(3), 'Sentiment'];
                                        return [value, name];
                                    }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="score"
                                    stroke="#0ea5e9"
                                    strokeWidth={2}
                                    dot={{ fill: '#0ea5e9', r: 3 }}
                                    activeDot={{ r: 6, onClick: (_: any, index: number) => setSelectedEntry(dataset.data[index]) }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* Distribution Pie Chart */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="card"
                    >
                        <h2 className="text-lg font-semibold text-neutral-900 mb-6">
                            Distribution
                        </h2>
                        <ResponsiveContainer width="100%" height={350}>
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="45%"
                                    labelLine={true}
                                    label={({ cx, cy, midAngle, innerRadius, outerRadius, name, percent }: any) => {
                                        const RADIAN = Math.PI / 180;
                                        const radius = outerRadius + 30;
                                        const x = cx + radius * Math.cos(-midAngle * RADIAN);
                                        const y = cy + radius * Math.sin(-midAngle * RADIAN);
                                        const pct = ((percent || 0) * 100).toFixed(0);
                                        
                                        return (
                                            <text 
                                                x={x} 
                                                y={y} 
                                                fill="#404040" 
                                                textAnchor={x > cx ? 'start' : 'end'} 
                                                dominantBaseline="central"
                                                style={{ fontSize: '14px', fontWeight: '600' }}
                                            >
                                                {`${name} ${pct}%`}
                                            </text>
                                        );
                                    }}
                                    outerRadius={90}
                                    innerRadius={0}
                                    fill="#8884d8"
                                    dataKey="value"
                                    paddingAngle={2}
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} stroke="#fff" strokeWidth={2} />
                                    ))}
                                </Pie>
                                <Tooltip 
                                    formatter={(value: any, name: any) => [value, name]}
                                    contentStyle={{
                                        backgroundColor: 'white',
                                        border: '1px solid #e5e5e5',
                                        borderRadius: '8px',
                                        padding: '8px 12px',
                                    }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                        <div className="mt-4 space-y-2">
                            {[
                                { name: 'Positive', value: dataset.statistics.sentiment_distribution.positive, color: '#16a34a' },
                                { name: 'Neutral', value: dataset.statistics.sentiment_distribution.neutral, color: '#737373' },
                                { name: 'Negative', value: dataset.statistics.sentiment_distribution.negative, color: '#dc2626' },
                            ].map((item) => (
                                <div key={item.name} className="flex items-center justify-between text-sm">
                                    <div className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: item.color }}
                                        />
                                        <span className="text-neutral-700">{item.name}</span>
                                    </div>
                                    <span className="font-medium text-neutral-900">{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                </div>

                {/* Data Table */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="card"
                >
                    <h2 className="text-lg font-semibold text-neutral-900 mb-6">Detailed Analysis</h2>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-neutral-200">
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">ID</th>
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">Text</th>
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">Sentiment</th>
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">Score</th>
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">Confidence</th>
                                    <th className="text-left py-3 px-4 text-sm font-medium text-neutral-600">Action</th>
                                </tr>
                            </thead>
                            <tbody>
                                {dataset.data.slice(0, 10).map((entry) => (
                                    <tr
                                        key={entry.id}
                                        className="border-b border-neutral-100 hover:bg-neutral-50 transition-colors"
                                    >
                                        <td className="py-3 px-4 text-sm text-neutral-700">{entry.id}</td>
                                        <td className="py-3 px-4 text-sm text-neutral-700 max-w-md truncate">
                                            {entry.text}
                                        </td>
                                        <td className="py-3 px-4">
                                            <span className={`badge ${getSentimentBadge(entry.sentiment_label)} flex items-center gap-1 w-fit`}>
                                                {getSentimentIcon(entry.sentiment_label)}
                                                {entry.sentiment_label}
                                            </span>
                                        </td>
                                        <td className={`py-3 px-4 text-sm font-medium ${getSentimentColor(entry.sentiment_label)}`}>
                                            {entry.sentiment_score.toFixed(3)}
                                        </td>
                                        <td className="py-3 px-4 text-sm text-neutral-700">
                                            {(entry.confidence * 100).toFixed(1)}%
                                        </td>
                                        <td className="py-3 px-4">
                                            <button
                                                onClick={() => setSelectedEntry(entry)}
                                                className="text-primary-600 hover:text-primary-700 text-sm font-medium transition-colors"
                                            >
                                                View Details
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    {dataset.data.length > 10 && (
                        <div className="mt-4 text-center text-sm text-neutral-600">
                            Showing 10 of {dataset.data.length} entries
                        </div>
                    )}
                </motion.div>
                    </>
                )}

                {/* AI Analytics Tab */}
                {activeTab === 'analytics' && (
                    <>
                        {/* Trends Overview */}
                        {dataset.trends && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
                            >
                                <div className="stat-card">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-medium text-neutral-600">Health Score</span>
                                        <Activity className="w-5 h-5 text-primary-600" />
                                    </div>
                                    <div className="text-3xl font-bold text-neutral-900">
                                        {dataset.trends.health_score.toFixed(0)}
                                    </div>
                                    <div className="text-xs text-neutral-500 mt-1">Overall Performance</div>
                                </div>

                                <div className="stat-card">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-medium text-neutral-600">Trend</span>
                                        {dataset.trends.trend_direction === 'improving' ? (
                                            <TrendingUp className="w-5 h-5 text-green-600" />
                                        ) : dataset.trends.trend_direction === 'declining' ? (
                                            <TrendingDown className="w-5 h-5 text-red-600" />
                                        ) : (
                                            <Minus className="w-5 h-5 text-neutral-600" />
                                        )}
                                    </div>
                                    <div className="text-2xl font-bold text-neutral-900 capitalize">
                                        {dataset.trends.trend_direction}
                                    </div>
                                    <div className="text-xs text-neutral-500 mt-1">{dataset.trends.momentum}</div>
                                </div>

                                <div className="stat-card">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-medium text-neutral-600">Volatility</span>
                                        <BarChart3 className="w-5 h-5 text-orange-600" />
                                    </div>
                                    <div className="text-3xl font-bold text-neutral-900">
                                        {(dataset.trends.volatility * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-xs text-neutral-500 mt-1">Experience Consistency</div>
                                </div>

                                <div className="stat-card">
                                    <div className="flex items-center justify-between mb-3">
                                        <span className="text-sm font-medium text-neutral-600">Confidence</span>
                                        <Eye className="w-5 h-5 text-primary-600" />
                                    </div>
                                    <div className="text-3xl font-bold text-neutral-900">
                                        {(dataset.trends.average_confidence * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-xs text-neutral-500 mt-1">Prediction Accuracy</div>
                                </div>
                            </motion.div>
                        )}

                        {/* AI Forecast Section */}
                        {dataset.forecast && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.2 }}
                                className="card mb-8"
                            >
                                <div className="flex items-center justify-between mb-6">
                                    <div>
                                        <h2 className="text-lg font-semibold text-neutral-900 flex items-center gap-2">
                                            <Activity className="w-5 h-5 text-primary-600" />
                                            AI-Powered Sentiment Forecast
                                        </h2>
                                        <p className="text-sm text-neutral-600 mt-1">{dataset.forecast.forecast_summary}</p>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm text-neutral-600">Forecast Confidence</div>
                                        <div className="text-2xl font-bold text-primary-600">{dataset.forecast.confidence_level.toFixed(0)}%</div>
                                    </div>
                                </div>
                                
                                <ResponsiveContainer width="100%" height={350}>
                                    <LineChart data={dataset.forecast.forecasts.map((value, index) => ({
                                        week: dataset.forecast!.time_labels[index],
                                        forecast: value,
                                        upper: dataset.forecast!.upper_bound[index],
                                        lower: dataset.forecast!.lower_bound[index]
                                    }))}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                        <XAxis
                                            dataKey="week"
                                            stroke="#737373"
                                            style={{ fontSize: '12px' }}
                                        />
                                        <YAxis
                                            stroke="#737373"
                                            style={{ fontSize: '12px' }}
                                            domain={[-1, 1]}
                                            label={{ value: 'Sentiment Score', angle: -90, position: 'insideLeft' }}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: 'white',
                                                border: '1px solid #e5e5e5',
                                                borderRadius: '8px',
                                                padding: '12px',
                                            }}
                                            formatter={(value: any) => value.toFixed(3)}
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="upper"
                                            stroke="#93c5fd"
                                            strokeWidth={1}
                                            strokeDasharray="5 5"
                                            dot={false}
                                            name="Upper Bound"
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="forecast"
                                            stroke="#0ea5e9"
                                            strokeWidth={3}
                                            dot={{ fill: '#0ea5e9', r: 4 }}
                                            name="Predicted Sentiment"
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="lower"
                                            stroke="#93c5fd"
                                            strokeWidth={1}
                                            strokeDasharray="5 5"
                                            dot={false}
                                            name="Lower Bound"
                                        />
                                    </LineChart>
                                </ResponsiveContainer>

                                <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                                    <h3 className="text-sm font-semibold text-blue-900 mb-2">About This Forecast</h3>
                                    <p className="text-xs text-blue-800">
                                        This AI-powered forecast uses time series analysis, trend detection, and historical patterns 
                                        to predict sentiment evolution over the next 12 weeks. The shaded area represents the confidence interval.
                                    </p>
                                </div>
                            </motion.div>
                        )}

                        {/* AI Recommendations */}
                        {dataset.recommendations && dataset.recommendations.length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 }}
                                className="card mb-8"
                            >
                                <h2 className="text-lg font-semibold text-neutral-900 mb-6 flex items-center gap-2">
                                    <Lightbulb className="w-5 h-5 text-yellow-600" />
                                    Strategic Recommendations
                                </h2>
                                
                                <div className="space-y-4">
                                    {dataset.recommendations.map((rec, index) => {
                                        const priorityColors = {
                                            high: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', icon: AlertTriangle },
                                            medium: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', icon: Info },
                                            low: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', icon: CheckCircle }
                                        };
                                        
                                        const colors = priorityColors[rec.priority];
                                        const IconComponent = colors.icon;
                                        
                                        return (
                                            <div
                                                key={index}
                                                className={`p-4 rounded-lg border-2 ${colors.border} ${colors.bg}`}
                                            >
                                                <div className="flex items-start gap-3">
                                                    <IconComponent className={`w-5 h-5 ${colors.text} mt-0.5 flex-shrink-0`} />
                                                    <div className="flex-1">
                                                        <div className="flex items-start justify-between mb-2">
                                                            <div>
                                                                <h3 className={`font-semibold ${colors.text}`}>{rec.title}</h3>
                                                                <span className="text-xs text-neutral-600">{rec.category}</span>
                                                            </div>
                                                            <span className={`badge ${colors.bg} ${colors.text} border-0 text-xs`}>
                                                                {rec.impact} Impact
                                                            </span>
                                                        </div>
                                                        <p className="text-sm text-neutral-700 mb-2">{rec.description}</p>
                                                        <div className="bg-white bg-opacity-50 p-3 rounded mt-2 border border-neutral-200">
                                                            <div className="text-xs font-medium text-neutral-600 mb-1 flex items-center gap-1">
                                                                <Target className="w-3 h-3" />
                                                                Recommended Action
                                                            </div>
                                                            <p className="text-sm text-neutral-800">{rec.action}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </motion.div>
                        )}
                    </>
                )}
            </div>

            {/* Detail Modal */}
            <AnimatePresence>
                {selectedEntry && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6 z-50"
                        onClick={() => setSelectedEntry(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-white rounded-xl shadow-strong max-w-3xl w-full max-h-[80vh] overflow-y-auto p-8"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="flex items-start justify-between mb-6">
                                <div>
                                    <h3 className="text-2xl font-bold text-neutral-900 mb-2">
                                        Detailed Analysis
                                    </h3>
                                    <span className={`badge ${getSentimentBadge(selectedEntry.sentiment_label)} flex items-center gap-1 w-fit`}>
                                        {getSentimentIcon(selectedEntry.sentiment_label)}
                                        {selectedEntry.sentiment_label}
                                    </span>
                                </div>
                                <button
                                    onClick={() => setSelectedEntry(null)}
                                    className="text-neutral-400 hover:text-neutral-600 transition-colors"
                                >
                                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>

                            <div className="space-y-6">
                                {/* Text with Attention Weights */}
                                <div>
                                    <h4 className="text-sm font-medium text-neutral-600 mb-3">
                                        Text with Attention Visualization
                                    </h4>
                                    <div className="p-4 bg-neutral-50 rounded-lg leading-relaxed">
                                        {selectedEntry.tokens.map((token, idx) => {
                                            const weight = selectedEntry.attention_weights[idx] || 0;
                                            const opacity = Math.min(weight * 3, 1);
                                            const isPositive = selectedEntry.sentiment_label === 'Positive';
                                            const isNegative = selectedEntry.sentiment_label === 'Negative';

                                            return (
                                                <span
                                                    key={idx}
                                                    className="inline-block px-1 py-0.5 mx-0.5 rounded transition-all"
                                                    style={{
                                                        backgroundColor: isPositive
                                                            ? `rgba(34, 197, 94, ${opacity * 0.3})`
                                                            : isNegative
                                                                ? `rgba(239, 68, 68, ${opacity * 0.3})`
                                                                : `rgba(115, 115, 115, ${opacity * 0.2})`,
                                                        fontWeight: weight > 0.1 ? 600 : 400,
                                                    }}
                                                    title={`Attention: ${weight.toFixed(3)}`}
                                                >
                                                    {token}
                                                </span>
                                            );
                                        })}
                                    </div>
                                    <p className="text-xs text-neutral-500 mt-2">
                                        Highlighted words have higher attention weights (more important for sentiment)
                                    </p>
                                </div>

                                {/* Metrics */}
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-neutral-50 rounded-lg">
                                        <div className="text-sm text-neutral-600 mb-1">Sentiment Score</div>
                                        <div className={`text-2xl font-bold ${getSentimentColor(selectedEntry.sentiment_label)}`}>
                                            {selectedEntry.sentiment_score.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="p-4 bg-neutral-50 rounded-lg">
                                        <div className="text-sm text-neutral-600 mb-1">Confidence</div>
                                        <div className="text-2xl font-bold text-neutral-900">
                                            {(selectedEntry.confidence * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>

                                {/* Probabilities */}
                                <div>
                                    <h4 className="text-sm font-medium text-neutral-600 mb-3">
                                        Class Probabilities
                                    </h4>
                                    <div className="space-y-3">
                                        {Object.entries(selectedEntry.probabilities).map(([key, value]) => (
                                            <div key={key}>
                                                <div className="flex items-center justify-between text-sm mb-1">
                                                    <span className="capitalize text-neutral-700">{key}</span>
                                                    <span className="font-medium text-neutral-900">
                                                        {((value as number) * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                                <div className="w-full bg-neutral-200 rounded-full h-2">
                                                    <div
                                                        className={`h-full rounded-full ${key === 'positive'
                                                            ? 'bg-green-500'
                                                            : key === 'negative'
                                                                ? 'bg-red-500'
                                                                : 'bg-neutral-500'
                                                            }`}
                                                        style={{ width: `${(value as number) * 100}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Original Text */}
                                <div>
                                    <h4 className="text-sm font-medium text-neutral-600 mb-3">Original Text</h4>
                                    <div className="p-4 bg-neutral-50 rounded-lg text-sm text-neutral-700">
                                        {selectedEntry.text}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Dashboard;
