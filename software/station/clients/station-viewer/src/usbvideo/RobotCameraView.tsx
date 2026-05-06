import { memo } from 'react';
import { Link } from 'react-router-dom';
import { st3215, usbvideo } from '../api/proto.js';
import MotorDataTable from '../st3215/MotorDataTable';
import { formatCameraName } from './camera-source';
import CameraViewer from './CameraViewer';

interface RobotCameraViewProps {
  primaryVideoSource?: usbvideo.IRxEnvelope;
  secondaryVideoSource?: usbvideo.IRxEnvelope;
  bus: st3215.InferenceState.IBusState;
  busIndex: number;
  isWebControlled?: boolean;
  showMotorData?: boolean;
  showCalibrateButton?: boolean;
  needsCalibration?: boolean;
}

const RobotCameraView = memo(function RobotCameraView({
  primaryVideoSource,
  secondaryVideoSource,
  bus,
  busIndex,
  isWebControlled,
  showMotorData = true,
  showCalibrateButton,
  needsCalibration,
}: RobotCameraViewProps) {
  const motorCount = bus.motors?.length ?? 0;
  // 8-motor humanoids need room for two arm groups without crowding the camera view.
  const motorPanelHeight =
    motorCount >= 8 ? 'clamp(240px, 40%, 320px)' : 'clamp(180px, 32%, 240px)';

  return (
    <div className="flex flex-col w-full h-full min-h-0 overflow-hidden bg-black rounded-b-lg">
      <div className="relative min-h-0" style={{ flex: '1 1 auto' }}>
        {primaryVideoSource ? (
          <CameraViewer
            inferenceState={primaryVideoSource}
            className="h-full w-full"
            imageClassName="select-none"
            fit="contain"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-surface-primary/30 p-6 text-center">
            <div className="text-accent-warning text-lg font-bold uppercase tracking-wide">
              No camera selected
            </div>
            <p className="max-w-md text-sm text-text-muted">
              Select an active USB video source in the title bar to switch this robot card into a camera-first operator view.
            </p>
          </div>
        )}

        {secondaryVideoSource && (
          <div className="absolute bottom-4 right-4 z-30 h-[160px] w-2/5 min-w-[220px] max-w-[360px] overflow-hidden rounded-lg border-2 border-border-default bg-surface-primary shadow-2xl">
            <CameraViewer
              inferenceState={secondaryVideoSource}
              className="h-full w-full"
              imageClassName="select-none"
              fit="contain"
              overlay="none"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-surface-secondary/70 px-2 py-1 text-xs font-mono text-text-label backdrop-blur-sm">
              {formatCameraName(secondaryVideoSource)}
            </div>
          </div>
        )}

        {showCalibrateButton && needsCalibration && (
          <div className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none">
            <Link
              to="/st3215-bus-calibration"
              state={{ bus }}
              className="pointer-events-auto px-6 py-3 rounded text-lg font-bold transition-colors bg-accent-success-bg text-text-primary hover:bg-accent-success-deep ring-4 ring-accent-success-deep/50"
            >
              Calibrate
            </Link>
          </div>
        )}
      </div>

      {/* Motor data section */}
      {showMotorData && motorCount > 0 && (
        <div
          className="relative min-h-0 border-t border-border-default bg-surface-base"
          style={{ flex: `0 0 ${motorPanelHeight}` }}
        >
          <MotorDataTable
            bus={bus}
            busIndex={busIndex}
            isWebControlled={isWebControlled}
            layout="panel"
          />
        </div>
      )}
    </div>
  );
});

export default RobotCameraView;
