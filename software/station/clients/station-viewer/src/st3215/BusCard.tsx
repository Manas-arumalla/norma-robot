import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Long from "long";
import { Camera } from "lucide-react";
import { Link } from "react-router-dom";
import { commandManager } from "../api/commands";
import { FrameEntry } from "../api/frame-parser";
import { motors_mirroring, st3215, usbvideo } from "../api/proto";
import { serverToLocal } from "../api/timestamp-utils";
import { getLatencyBgColor, getLatencyTextColor } from "@/utils/color-utils";
import { getVideoSourceId, getVideoSourceLabel } from "../usbvideo/camera-source";
import CameraViewer from "../usbvideo/CameraViewer";
import RobotCameraView from "../usbvideo/RobotCameraView";
import BusWebGLRenderer from "./BusWebGLRenderer";
import MotorDataTable from "./MotorDataTable";
import { ADDR_GOAL_POSITION, getMotorPosition } from "./motor-parser";

interface LatencyReading {
  timestamp: number;
  latency: number;
}

interface LatencyStats {
  avg: number;
  min: number;
  max: number;
}

const STALE_CAMERA_MAX_AGE_MS = 60_000;
const MIN_CALIBRATED_RANGE = 100;

type RobotViewMode = "model" | "camera";

interface BusCardProps {
  bus: st3215.InferenceState.IBusState;
  busIndex: number;
  videoSources?: FrameEntry<usbvideo.IRxEnvelope>[];
  allBuses?: st3215.InferenceState.IBusState[] | null;
  mirroringState?: motors_mirroring.IInferenceState;
}

const BusCard: React.FC<BusCardProps> = ({
  bus,
  busIndex,
  videoSources,
  allBuses,
  mirroringState,
}) => {
  const latencyHistoryRef = useRef<Map<string, LatencyReading[]>>(new Map());
  const hasPrimaryVideoSourcePreferenceRef = useRef(false);
  const [primaryVideoSourceId, setPrimaryVideoSourceId] = useState<string | null>(null);
  const [secondaryVideoSourceId, setSecondaryVideoSourceId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<RobotViewMode>("model");
  const [isWebControlled, setIsWebControlled] = useState(false);

  const activeVideoSources = useMemo(() => {
    if (!videoSources) {
      return [];
    }

    const nowMs = Date.now();

    return videoSources.filter((entry) => {
      const monotonicStampNs = entry.data.stamp?.monotonicStampNs;
      if (!monotonicStampNs) {
        return true;
      }

      const localStampNs = serverToLocal(Long.fromValue(monotonicStampNs));
      const ageMs = nowMs - localStampNs.toNumber() / 1e6;

      return ageMs <= STALE_CAMERA_MAX_AGE_MS;
    });
  }, [videoSources]);

  const primaryVideoSource = activeVideoSources.find(
    (entry) => getVideoSourceId(entry) === primaryVideoSourceId,
  )?.data;

  const secondaryVideoSource = activeVideoSources.find(
    (entry) => getVideoSourceId(entry) === secondaryVideoSourceId,
  )?.data;

  const activeVideoSourceIds = useMemo(
    () => activeVideoSources.map(getVideoSourceId),
    [activeVideoSources],
  );
  const firstActiveVideoSourceId = activeVideoSourceIds[0] ?? null;

  useEffect(() => {
    if (primaryVideoSourceId && !activeVideoSourceIds.includes(primaryVideoSourceId)) {
      setPrimaryVideoSourceId(null);
      hasPrimaryVideoSourcePreferenceRef.current = false;
    }

    if (
      !primaryVideoSourceId &&
      firstActiveVideoSourceId &&
      !hasPrimaryVideoSourcePreferenceRef.current
    ) {
      setPrimaryVideoSourceId(firstActiveVideoSourceId);
    }

    if (
      secondaryVideoSourceId &&
      (!activeVideoSourceIds.includes(secondaryVideoSourceId) || secondaryVideoSourceId === primaryVideoSourceId)
    ) {
      setSecondaryVideoSourceId(null);
    }
  }, [
    activeVideoSourceIds,
    firstActiveVideoSourceId,
    primaryVideoSourceId,
    secondaryVideoSourceId,
  ]);

  const handlePrimaryVideoSourceChange = useCallback((sourceId: string | null) => {
    // Keep this sticky even for "None" so an explicit user clear is not auto-filled again.
    hasPrimaryVideoSourcePreferenceRef.current = true;
    setPrimaryVideoSourceId(sourceId);
    if (sourceId && sourceId === secondaryVideoSourceId) {
      setSecondaryVideoSourceId(null);
    }
  }, [secondaryVideoSourceId]);

  const handleSecondaryVideoSourceChange = useCallback((sourceId: string | null) => {
    if (sourceId && sourceId === primaryVideoSourceId) {
      return;
    }

    setSecondaryVideoSourceId(sourceId);
  }, [primaryVideoSourceId]);

  const handleControlSourceChange = async (sourceBusSerial: string | null) => {
    if (!bus.bus?.serialNumber) {
      return;
    }

    // Handle Web-controlled mode
    if (sourceBusSerial === "web-controlled") {
      const target: motors_mirroring.IMirroringBus = {
        type: motors_mirroring.BusType.MBT_ST3215,
        uniqueId: bus.bus.serialNumber,
      };

      // Stop any existing mirroring
      await commandManager.sendMirroringCommand({
        type: motors_mirroring.CommandType.CT_STOP_MIRROR,
        source: target,
      });

      // Freeze all motors by sending their current positions
      if (bus.motors) {
        const commands = [];
        for (const motor of bus.motors) {
          if (motor.id !== null && motor.id !== undefined && motor.state) {
            const currentPosition = getMotorPosition(motor.state);

            // Send command to set motor to its current position (freeze it)
            const command = st3215.Command.create({
              targetBusSerial: bus.bus?.serialNumber,
              write: {
                motorId: motor.id,
                address: ADDR_GOAL_POSITION,
                value: new Uint8Array([
                  currentPosition & 0xff,
                  (currentPosition >> 8) & 0xff,
                ]),
              },
            });
            commands.push(command);
          }
        }
        if (commands.length > 0) {
          await commandManager.sendSt3215Commands(commands);
        }
      }

      setIsWebControlled(true);
      return;
    }

    setIsWebControlled(false);

    const target: motors_mirroring.IMirroringBus = {
      type: motors_mirroring.BusType.MBT_ST3215,
      uniqueId: bus.bus.serialNumber,
    };

    if (sourceBusSerial) {
      const source: motors_mirroring.IMirroringBus = {
        type: motors_mirroring.BusType.MBT_ST3215,
        uniqueId: sourceBusSerial,
      };
      await commandManager.sendMirroringCommand({
        type: motors_mirroring.CommandType.CT_START_MIRROR,
        source: source,
        targets: [target],
      });
    } else {
      await commandManager.sendMirroringCommand({
        type: motors_mirroring.CommandType.CT_STOP_MIRROR,
        source: target,
      });
    }
  };

  const currentMirror = mirroringState?.mirroring?.find((m) =>
    m.targets?.some((t) => t.id?.uniqueId === bus.bus?.serialNumber),
  );

  // Function to calculate moving average for latency (15 second window)
  const getMovingAverageLatency = (
    key: string,
    currentLatency: number,
  ): LatencyStats => {
    const now = Date.now();
    // Clamp to prevent negative values
    const validLatency = Math.max(0, currentLatency);

    const history = latencyHistoryRef.current.get(key) || [];

    // Add current reading
    history.push({ timestamp: now, latency: validLatency });

    // Filter to keep only last 15 seconds
    const filtered = history.filter((h) => now - h.timestamp <= 15000);
    latencyHistoryRef.current.set(key, filtered);

    // Calculate statistics
    if (filtered.length === 0) {
      return { avg: validLatency, min: validLatency, max: validLatency };
    }

    const latencies = filtered.map((h) => h.latency);
    const sum = latencies.reduce((acc, l) => acc + l, 0);

    return {
      avg: sum / filtered.length,
      min: Math.min(...latencies),
      max: Math.max(...latencies),
    };
  };

  const adjustedBusStamp = bus.monotonicStampNs
    ? serverToLocal(Long.fromValue(bus.monotonicStampNs))
    : null;
  const now = Date.now();
  const busLatency = adjustedBusStamp
    ? now - adjustedBusStamp.toNumber() / 1e6
    : 0;
  const busLatencyAvg = getMovingAverageLatency(`bus-${busIndex}`, busLatency);

  const hasMotors = (bus.motors?.length ?? 0) > 0;
  const hasUnfrozenMotor =
    hasMotors && bus.motors!.some((motor) => motor.rangeFreezed !== true);
  const hasNarrowRange =
    hasMotors &&
    bus.motors!.some(
      (motor) =>
        (motor.rangeMax ?? 0) - (motor.rangeMin ?? 0) < MIN_CALIBRATED_RANGE,
    );
  const needsCalibration = hasMotors && (hasUnfrozenMotor || hasNarrowRange);
  const canRender3d = [6, 8].includes(bus.motors?.length || 0);
  const canShowCamera = activeVideoSources.length > 0;
  const controlSourceWidthClass = viewMode === "camera" ? "max-w-[140px]" : "max-w-[180px]";
  const cameraSelectWidthClass = viewMode === "camera" ? "max-w-[120px]" : "max-w-[180px]";

  return (
    <div className="min-w-0 border border-border-default rounded-lg bg-surface-primary/50">
      {/* Title Bar */}
      <div className="bg-surface-secondary/50 px-4 py-2 rounded-t-lg flex flex-col gap-2 border-b border-border-default items-start">
        <div className="flex w-full flex-wrap items-center gap-2">
          <span className="font-bold text-lg text-accent-data">
            #{bus.bus?.serialNumber}
          </span>
          <div className="flex rounded-md border border-border-subtle bg-surface-primary p-0.5">
            <button
              type="button"
              onClick={() => setViewMode("model")}
              className={`flex h-8 min-w-8 items-center justify-center rounded px-2 text-xs font-bold transition-colors ${viewMode === "model" ? "bg-accent-success-bg text-text-primary" : "text-text-muted hover:text-text-primary"}`}
              title="3D view"
              aria-label="3D view"
            >
              3D
            </button>
            <button
              type="button"
              onClick={() => setViewMode("camera")}
              disabled={!canShowCamera}
              className={`flex h-8 w-8 items-center justify-center rounded transition-colors disabled:cursor-not-allowed disabled:text-text-muted ${viewMode === "camera" ? "bg-accent-data text-surface-base" : "text-text-muted hover:text-text-primary"}`}
              title={canShowCamera ? "Camera-first robot view" : "No active cameras"}
              aria-label="Camera view"
            >
              <Camera className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
          <select
            value={
              isWebControlled
                ? "web-controlled"
                : (currentMirror?.source?.id?.uniqueId ?? "")
            }
            onChange={(e) => handleControlSourceChange(e.target.value || null)}
            className={`block min-w-0 ${controlSourceWidthClass} pl-3 pr-10 py-1 text-base border-border-subtle bg-surface-secondary text-text-primary focus:outline-none focus:ring-accent-success-deep focus:border-accent-success-deep sm:text-sm rounded-md`}
          >
            <option value="">(Self-controlled)</option>
            <option value="web-controlled">(Web-controlled)</option>
            {allBuses?.map((sourceBus) => {
              if (
                !sourceBus.bus?.serialNumber ||
                sourceBus.bus.serialNumber === bus.bus?.serialNumber
              ) {
                return null;
              }
              return (
                <option
                  key={sourceBus.bus.serialNumber}
                  value={sourceBus.bus.serialNumber}
                  title={`#${sourceBus.bus.serialNumber}`}
                >
                  #{sourceBus.bus.serialNumber}
                </option>
              );
            })}
          </select>
          <select
            value={primaryVideoSourceId ?? ""}
            onChange={(e) => handlePrimaryVideoSourceChange(e.target.value || null)}
            className={`block min-w-0 ${cameraSelectWidthClass} pl-3 pr-10 py-1 text-base border-border-subtle bg-surface-secondary text-text-primary focus:outline-none focus:ring-accent-success-deep focus:border-accent-success-deep sm:text-sm rounded-md`}
            title="Main camera"
          >
            <option value="">None</option>
            {activeVideoSources
              .filter((entry) => getVideoSourceId(entry) !== secondaryVideoSourceId)
              .map((entry) => {
                const sourceId = getVideoSourceId(entry);
                const label = getVideoSourceLabel(entry);
                return (
                  <option
                    key={`${entry.queueId}-${sourceId}`}
                    value={sourceId}
                    title={label}
                  >
                    {label}
                  </option>
                );
              })}
          </select>
          {viewMode === "camera" && (
            <select
              value={secondaryVideoSourceId ?? ""}
              onChange={(e) => handleSecondaryVideoSourceChange(e.target.value || null)}
              className={`block min-w-0 ${cameraSelectWidthClass} basis-full pl-3 pr-10 py-1 text-base border-border-subtle bg-surface-secondary text-text-primary focus:outline-none focus:ring-accent-success-deep focus:border-accent-success-deep sm:text-sm rounded-md`}
              title="Picture-in-picture camera"
            >
              <option value="">None</option>
              {activeVideoSources
                .filter((entry) => getVideoSourceId(entry) !== primaryVideoSourceId)
                .map((entry) => {
                  const sourceId = getVideoSourceId(entry);
                  const label = getVideoSourceLabel(entry);
                  return (
                    <option
                      key={`${entry.queueId}-${sourceId}`}
                      value={sourceId}
                      title={label}
                    >
                      {label}
                    </option>
                  );
                })}
            </select>
          )}
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5">
            <span className="text-text-muted">Port:</span>
            <span className="text-accent-data">{bus.bus?.portName || "N/A"}</span>
          </div>
          <span className={`${getLatencyTextColor(busLatency)}`}>
            {busLatencyAvg.avg < 1000
              ? `${busLatencyAvg.avg.toFixed(0)}ms`
              : `${(busLatencyAvg.avg / 1000).toFixed(1)}s`}
          </span>
          <span
            className={`w-3 h-3 rounded-full ${getLatencyBgColor(busLatency, false)}`}
          ></span>
        </div>
      </div>

      {/* Content */}
      <div className="relative h-180">
        {viewMode === "camera" ? (
          <RobotCameraView
            primaryVideoSource={primaryVideoSource}
            secondaryVideoSource={secondaryVideoSource}
            bus={bus}
            busIndex={busIndex}
            showMotorData={true}
            showCalibrateButton={true}
            needsCalibration={needsCalibration}
            isWebControlled={isWebControlled}
          />
        ) : canRender3d ? (
          <BusWebGLRenderer
            busSerialNumber={bus.bus?.serialNumber}
            bus={bus}
            busIndex={busIndex}
            showMotorData={true}
            selectedVideoSource={primaryVideoSource}
            showCalibrateButton={true}
            needsCalibration={needsCalibration}
            isWebControlled={isWebControlled}
          />
        ) : (
          <>
            <div className="absolute inset-0 p-4 flex flex-col items-center justify-center bg-surface-primary/20">
              <p className="text-accent-warning mb-4 text-center">
                {(bus.motors?.length || 0) === 0 ? (
                  <>No motors connected to this bus.</>
                ) : (
                  <>
                    3D model visualization is only available for 6 or 8-motor
                    configurations.
                    <br />
                    This bus has {bus.motors?.length} motor
                    {bus.motors?.length === 1 ? "" : "s"}.
                  </>
                )}
              </p>
              <div className="flex gap-4">
                {(bus.motors?.length || 0) > 1 && (
                  <Link
                    to="/st3215-bus-calibration"
                    state={{ bus }}
                    className={`px-4 py-2 rounded text-base font-bold transition-colors bg-accent-success-bg text-text-primary hover:bg-accent-success-deep ${needsCalibration ? "ring-4 ring-accent-success-deep/50 scale-110" : ""}`}
                  >
                    Calibrate
                  </Link>
                )}
                {bus.motors?.length === 1 && (
                  <Link
                    to={`/st3215-bind-motors`}
                    state={{ bus }}
                    className="bg-accent-info-bg hover:bg-accent-info-deep px-4 py-2 rounded text-text-primary transition-colors"
                    title="Configure motor ID"
                  >
                    Configure Motor ID
                  </Link>
                )}
              </div>
            </div>
            <div className="absolute inset-0 pointer-events-none">
              <div className="pointer-events-auto">
                <MotorDataTable
                  bus={bus}
                  busIndex={busIndex}
                  isWebControlled={isWebControlled}
                />
              </div>
            </div>
            {primaryVideoSource && (
              <div className="absolute top-4 right-4 h-[200px] w-2/5 max-w-[520px] overflow-hidden rounded-lg border border-border-default bg-black shadow-lg pointer-events-auto">
                <CameraViewer inferenceState={primaryVideoSource} className="h-full w-full" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default BusCard;
