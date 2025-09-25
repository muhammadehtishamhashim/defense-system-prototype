import React, { useState } from 'react';
import Button from '../ui/Button';
import Modal from '../ui/Modal';
import { 
  MagnifyingGlassPlusIcon,
  ArrowDownTrayIcon,
  XMarkIcon,
  ChevronLeftIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';

interface Snapshot {
  id: string;
  url: string;
  timestamp: number;
  description?: string;
  boundingBoxes?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    label?: string;
    confidence?: number;
  }>;
}

interface SnapshotGalleryProps {
  snapshots: Snapshot[];
  onSnapshotSelect?: (snapshot: Snapshot) => void;
  onDownload?: (snapshot: Snapshot) => void;
  className?: string;
}

const SnapshotGallery: React.FC<SnapshotGalleryProps> = ({
  snapshots,
  onSnapshotSelect,
  onDownload,
  className = ''
}) => {
  const [selectedSnapshot, setSelectedSnapshot] = useState<Snapshot | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleSnapshotClick = (snapshot: Snapshot, index: number) => {
    setSelectedSnapshot(snapshot);
    setCurrentIndex(index);
    setShowModal(true);
    onSnapshotSelect?.(snapshot);
  };

  const handlePrevious = () => {
    const newIndex = currentIndex > 0 ? currentIndex - 1 : snapshots.length - 1;
    setCurrentIndex(newIndex);
    setSelectedSnapshot(snapshots[newIndex]);
  };

  const handleNext = () => {
    const newIndex = currentIndex < snapshots.length - 1 ? currentIndex + 1 : 0;
    setCurrentIndex(newIndex);
    setSelectedSnapshot(snapshots[newIndex]);
  };

  const handleDownload = (snapshot: Snapshot) => {
    if (onDownload) {
      onDownload(snapshot);
    } else {
      // Default download behavior
      const link = document.createElement('a');
      link.href = snapshot.url;
      link.download = `snapshot_${snapshot.id}_${snapshot.timestamp}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  if (snapshots.length === 0) {
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <div className="text-lg mb-2">No snapshots available</div>
        <div className="text-sm">Snapshots will appear here when video alerts are generated</div>
      </div>
    );
  }

  return (
    <div className={className}>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {snapshots.map((snapshot, index) => (
          <div
            key={snapshot.id}
            className="relative group cursor-pointer bg-gray-100 rounded-lg overflow-hidden aspect-video"
            onClick={() => handleSnapshotClick(snapshot, index)}
          >
            <img
              src={snapshot.url}
              alt={`Snapshot ${snapshot.id}`}
              className="w-full h-full object-cover transition-transform group-hover:scale-105"
              onError={(e) => {
                (e.target as HTMLImageElement).src = '/placeholder-image.png';
              }}
            />
            
            {/* Overlay */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors">
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownload(snapshot);
                  }}
                  className="bg-black/50 text-white hover:bg-black/70"
                >
                  <ArrowDownTrayIcon className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="absolute bottom-2 left-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="bg-black/70 text-white text-xs p-2 rounded">
                  <div className="font-medium">
                    {formatTimestamp(snapshot.timestamp)}
                  </div>
                  {snapshot.description && (
                    <div className="mt-1 text-gray-300">
                      {snapshot.description}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Zoom icon */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
              <MagnifyingGlassPlusIcon className="h-8 w-8 text-white drop-shadow-lg" />
            </div>
          </div>
        ))}
      </div>

      {/* Modal for detailed view */}
      <Modal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        title="Snapshot Details"
        size="xl"
      >
        {selectedSnapshot && (
          <div className="space-y-4">
            <div className="relative">
              <img
                src={selectedSnapshot.url}
                alt={`Snapshot ${selectedSnapshot.id}`}
                className="w-full h-auto rounded-lg"
              />
              
              {/* Navigation buttons */}
              {snapshots.length > 1 && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handlePrevious}
                    className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-black/50 text-white hover:bg-black/70"
                  >
                    <ChevronLeftIcon className="h-5 w-5" />
                  </Button>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleNext}
                    className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-black/50 text-white hover:bg-black/70"
                  >
                    <ChevronRightIcon className="h-5 w-5" />
                  </Button>
                </>
              )}
            </div>

            {/* Snapshot info */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Snapshot Information</h4>
                  <div className="space-y-1 text-sm">
                    <div><span className="font-medium">ID:</span> {selectedSnapshot.id}</div>
                    <div><span className="font-medium">Timestamp:</span> {formatTimestamp(selectedSnapshot.timestamp)}</div>
                    {selectedSnapshot.description && (
                      <div><span className="font-medium">Description:</span> {selectedSnapshot.description}</div>
                    )}
                  </div>
                </div>
                
                {selectedSnapshot.boundingBoxes && selectedSnapshot.boundingBoxes.length > 0 && (
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Detected Objects</h4>
                    <div className="space-y-1 text-sm">
                      {selectedSnapshot.boundingBoxes.map((box, index) => (
                        <div key={index}>
                          <span className="font-medium">{box.label || `Object ${index + 1}`}:</span>
                          {box.confidence && ` ${Math.round(box.confidence * 100)}%`}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-500">
                {currentIndex + 1} of {snapshots.length}
              </div>
              
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => handleDownload(selectedSnapshot)}
                  className="flex items-center space-x-2"
                >
                  <ArrowDownTrayIcon className="h-4 w-4" />
                  <span>Download</span>
                </Button>
                
                <Button
                  variant="ghost"
                  onClick={() => setShowModal(false)}
                >
                  <XMarkIcon className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default SnapshotGallery;