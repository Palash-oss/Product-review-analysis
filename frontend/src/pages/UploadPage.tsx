import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, FileSpreadsheet, File, Loader2 } from 'lucide-react';
import { uploadFile } from '../api/client';
import type { ProcessedDataset } from '../types';

interface UploadPageProps {
    onUploadComplete: (dataset: ProcessedDataset) => void;
}

const UploadPage: React.FC<UploadPageProps> = ({ onUploadComplete }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            await handleFileUpload(files[0]);
        }
    }, []);

    const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            await handleFileUpload(files[0]);
        }
    }, []);

    const handleFileUpload = async (file: File) => {
        const validExtensions = ['.csv', '.xlsx', '.xls', '.pdf'];
        const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

        if (!validExtensions.includes(fileExtension)) {
            setError('Please upload a CSV, Excel, or PDF file.');
            return;
        }

        setError(null);
        setIsUploading(true);
        setUploadProgress(0);

        try {
            // Simulate progress animation
            const progressInterval = setInterval(() => {
                setUploadProgress((prev) => {
                    if (prev >= 90) {
                        clearInterval(progressInterval);
                        return 90;
                    }
                    return prev + 10;
                });
            }, 200);

            const dataset = await uploadFile(file);

            clearInterval(progressInterval);
            setUploadProgress(100);

            // Wait for animation to complete
            setTimeout(() => {
                onUploadComplete(dataset);
            }, 500);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to upload file. Please try again.');
            setIsUploading(false);
            setUploadProgress(0);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-6">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="w-full max-w-2xl"
            >
                {/* Header */}
                <div className="text-center mb-12">
                    <motion.h1
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="text-4xl font-bold text-neutral-900 mb-3"
                    >
                        Sentiment Analysis Platform
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                        className="text-neutral-600 text-lg"
                    >
                        Upload your data to begin advanced sentiment analysis
                    </motion.p>
                </div>

                {/* Upload Zone */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4 }}
                    className="card"
                >
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`
              relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300
              ${isDragging
                                ? 'border-primary-500 bg-primary-50'
                                : 'border-neutral-200 hover:border-primary-300 hover:bg-neutral-50'
                            }
              ${isUploading ? 'pointer-events-none' : 'cursor-pointer'}
            `}
                    >
                        <input
                            type="file"
                            accept=".csv,.xlsx,.xls,.pdf"
                            onChange={handleFileSelect}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            disabled={isUploading}
                        />

                        {!isUploading ? (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="space-y-4"
                            >
                                <div className="flex justify-center">
                                    <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center">
                                        <Upload className="w-8 h-8 text-primary-600" />
                                    </div>
                                </div>

                                <div>
                                    <p className="text-lg font-medium text-neutral-900 mb-2">
                                        Drop your file here or click to browse
                                    </p>
                                    <p className="text-sm text-neutral-500">
                                        Supports CSV, Excel (.xlsx, .xls), and PDF files
                                    </p>
                                </div>

                                <div className="flex items-center justify-center gap-6 pt-4">
                                    <div className="flex items-center gap-2 text-neutral-600">
                                        <FileText className="w-5 h-5" />
                                        <span className="text-sm">CSV</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-neutral-600">
                                        <FileSpreadsheet className="w-5 h-5" />
                                        <span className="text-sm">Excel</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-neutral-600">
                                        <File className="w-5 h-5" />
                                        <span className="text-sm">PDF</span>
                                    </div>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="space-y-6"
                            >
                                <div className="flex justify-center">
                                    <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
                                </div>

                                <div>
                                    <p className="text-lg font-medium text-neutral-900 mb-2">
                                        Processing your file...
                                    </p>
                                    <p className="text-sm text-neutral-500 mb-4">
                                        Analyzing sentiment and extracting insights
                                    </p>

                                    {/* Progress Bar */}
                                    <div className="w-full bg-neutral-200 rounded-full h-2.5 overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${uploadProgress}%` }}
                                            transition={{ duration: 0.3, ease: "easeOut" }}
                                            className="h-full bg-gradient-to-r from-green-500 via-green-600 to-green-500 shadow-lg"
                                            style={{
                                                boxShadow: '0 0 10px rgba(34, 197, 94, 0.5)'
                                            }}
                                        />
                                    </div>
                                    <p className="text-xs text-neutral-500 mt-2 font-medium">{uploadProgress}%</p>
                                </div>

                                {/* Animated Waveform */}
                                <div className="flex items-center justify-center gap-1 h-12">
                                    {[...Array(20)].map((_, i) => (
                                        <motion.div
                                            key={i}
                                            className="w-1 bg-primary-400 rounded-full"
                                            animate={{
                                                height: [
                                                    Math.random() * 30 + 10,
                                                    Math.random() * 40 + 20,
                                                    Math.random() * 30 + 10,
                                                ],
                                            }}
                                            transition={{
                                                duration: 0.8,
                                                repeat: Infinity,
                                                delay: i * 0.05,
                                            }}
                                        />
                                    ))}
                                </div>
                            </motion.div>
                        )}
                    </div>

                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg"
                        >
                            <p className="text-sm text-red-700">{error}</p>
                        </motion.div>
                    )}
                </motion.div>

                {/* Info Cards */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="grid grid-cols-3 gap-4 mt-8"
                >
                    <div className="text-center p-4">
                        <div className="text-2xl font-bold text-primary-600 mb-1">AI-Powered</div>
                        <div className="text-sm text-neutral-600">Deep Learning Model</div>
                    </div>
                    <div className="text-center p-4">
                        <div className="text-2xl font-bold text-primary-600 mb-1">Explainable</div>
                        <div className="text-sm text-neutral-600">Attention Visualization</div>
                    </div>
                    <div className="text-center p-4">
                        <div className="text-2xl font-bold text-primary-600 mb-1">Real-time</div>
                        <div className="text-sm text-neutral-600">Instant Analysis</div>
                    </div>
                </motion.div>
            </motion.div>
        </div>
    );
};

export default UploadPage;
