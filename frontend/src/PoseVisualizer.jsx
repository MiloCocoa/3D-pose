import { useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line, Html, Grid } from '@react-three/drei';
import * as THREE from 'three';

// Bone connections (same as before)
const BONE_LIST = [
  [0, 1], [1, 2], [1, 5], [1, 8],
  [2, 3], [3, 4], [5, 6], [6, 7],
  [8, 9], [8, 12], [9, 10], [10, 11],
  [11, 17], [11, 18], [12, 13], [13, 14],
  [14, 15]
];

// --- Helper: Transform data for a specific frame ---

// Helper to ground the skeleton to the feet
function groundFrame(frame) {
  // frame is array of [x, y, z]
  if (!frame || frame.length === 0) return frame;

  // Define the 3 points for the plane:
  // Left Heel (17), Left Toe (18), Right Toe (15)
  const idxLHeel = 17;
  const idxLToe = 18;
  const idxRToe = 15;

  if (!frame[idxLHeel] || !frame[idxLToe] || !frame[idxRToe]) {
    return frame;
  }

  const pLHeel = new THREE.Vector3(...frame[idxLHeel]);
  const pLToe = new THREE.Vector3(...frame[idxLToe]);
  const pRToe = new THREE.Vector3(...frame[idxRToe]);

  // 1. Compute Plane Normal
  // Vector from LHeel to LToe
  const v1 = new THREE.Vector3().subVectors(pLToe, pLHeel);
  // Vector from LHeel to RToe
  const v2 = new THREE.Vector3().subVectors(pRToe, pLHeel);
  
  const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();

  // Ensure normal points UP (Positive Z in input data)
  if (normal.z < 0) normal.negate();

  // 2. Compute Rotation to align Normal to Global Z (0,0,1)
  const targetUp = new THREE.Vector3(0, 0, 1);
  const quaternion = new THREE.Quaternion().setFromUnitVectors(normal, targetUp);

  // 3. Compute Centroid (Anchor Point) of these 3 points
  const centroid = new THREE.Vector3()
    .add(pLHeel).add(pLToe).add(pRToe).divideScalar(3);

  // 4. Apply Transform to all points
  return frame.map(p => {
    const v = new THREE.Vector3(...p);
    
    // Translate to center the base at 0,0,0
    v.sub(centroid);
    
    // Rotate to flatten the floor plane
    v.applyQuaternion(quaternion);
    
    return [v.x, v.y, v.z];
  });
}

function getJointsFromInput(data, frameIndex) {
  // inputPoseData format: [frames][joints][3]
  if (!data || !data[frameIndex]) return [];
  
  const groundedFrame = groundFrame(data[frameIndex]);
  
  // Map Z-up data to Y-up scene: (x, z, -y)
  return groundedFrame.map(p => new THREE.Vector3(p[0], p[2], -p[1]));
}

function getJointsFromCorrected(data, frameIndex) {
  // correctedPoseData format: [57][frames]
  if (!data || data.length !== 57) return [];
  
  // Reconstruct frame [joints][3]
  const frame = [];
  for (let i = 0; i < 19; i++) {
    frame.push([
      data[i][frameIndex],
      data[i + 19][frameIndex],
      data[i + 38][frameIndex]
    ]);
  }
  
  const groundedFrame = groundFrame(frame);
  
  // Map Z-up data to Y-up scene: (x, z, -y)
  return groundedFrame.map(p => new THREE.Vector3(p[0], p[2], -p[1]));
}

// --- Component: Single Skeleton ---

function Skeleton({ joints, color, opacity = 1, position = [0, 0, 0], label }) {
  if (!joints.length) return null;

  return (
    <group position={position}>
      {/* Label */}
      <Html position={[0, 1.6, 0]} center transform>
        <div style={{ 
          color: color, 
          fontFamily: 'sans-serif', 
          fontWeight: 'bold',
          background: 'rgba(255,255,255,0.8)',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '12px'
        }}>
          {label}
        </div>
      </Html>

      {/* Joints */}
      {joints.map((v, i) => (
        <mesh key={i} position={v}>
          <sphereGeometry args={[0.025, 16, 16]} />
          <meshStandardMaterial color={color} transparent opacity={opacity} />
        </mesh>
      ))}

      {/* Bones */}
      {BONE_LIST.map(([startIdx, endIdx], i) => {
        const start = joints[startIdx];
        const end = joints[endIdx];
        return (
          <Line
            key={i}
            points={[start, end]}
            color={color}
            lineWidth={3}
            transparent
            opacity={opacity}
          />
        );
      })}
    </group>
  );
}

// --- Main Component ---

export function PoseVisualizer({ inputPoseData, correctedPoseData }) {
  const [frameIndex, setFrameIndex] = useState(0);
  const numFrames = inputPoseData?.length || 0;

  // Extract skeletons for current frame
  const inputJoints = useMemo(() => 
    getJointsFromInput(inputPoseData, frameIndex), 
  [inputPoseData, frameIndex]);

  const correctedJoints = useMemo(() => 
    getJointsFromCorrected(correctedPoseData, frameIndex), 
  [correctedPoseData, frameIndex]);

  return (
    <div className="visualizer">
      <div className="canvas-container" style={{ height: '500px', background: '#0f172a', borderRadius: '0.5rem', overflow: 'hidden' }}>
        <Canvas camera={{ position: [0, 1, 4], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          
          <Grid infiniteGrid fadeDistance={20} fadeStrength={5} />
          
          {/* Input Skeleton (Red, Left) */}
          <Skeleton 
            joints={inputJoints} 
            color="#ef4444" 
            position={[-0.8, -0.0, 0.5]} 
            label="Input"
          />

          {/* Target Skeleton (Blue, Right) */}
          <Skeleton 
            joints={correctedJoints} 
            color="#3b82f6" 
            position={[0.8, -0.0, 0.5]} 
            label="Target"
          />

          <OrbitControls target={[0, 0, 0]} />
        </Canvas>
      </div>

      <div className="visualizer__controls">
        <button 
          className="control-btn"
          onClick={() => setFrameIndex(f => Math.max(0, f - 1))}
        >
          ←
        </button>
        
        <input
          type="range"
          min="0"
          max={numFrames - 1}
          value={frameIndex}
          onChange={(e) => setFrameIndex(parseInt(e.target.value))}
          style={{ flex: 1 }}
        />
        
        <button 
          className="control-btn"
          onClick={() => setFrameIndex(f => Math.min(numFrames - 1, f + 1))}
        >
          →
        </button>
        
        <span style={{ minWidth: '4rem', textAlign: 'right' }}>
          {frameIndex} / {numFrames - 1}
        </span>
      </div>
    </div>
  );
}