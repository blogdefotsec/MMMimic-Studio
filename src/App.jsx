import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { CCDIKSolver } from 'three/examples/jsm/animation/CCDIKSolver.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

// ==========================================
// 0. 内置超级 URDF 引擎
// ==========================================
const MOCK_URDF_XML = `
<robot name="Placeholder_Bot">
  <link name="base_link" />
  <joint name="Shoulder" type="revolute">
    <parent link="base_link" /><child link="upper_arm" />
    <origin xyz="0 0 0" rpy="0 0 0" /><axis xyz="0 0 1" />
    <limit lower="-1.57" upper="1.57" />
  </joint>
  <link name="upper_arm" />
  <joint name="Elbow" type="revolute">
    <parent link="upper_arm" /><child link="forearm" />
    <origin xyz="0 2 0" rpy="0 0 0" /><axis xyz="0 0 1" />
    <limit lower="-2.5" upper="0" /> 
  </joint>
  <link name="forearm" />
</robot>
`;

function parseURDF(xmlString) {
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlString, "text/xml");
  const robot = { name: xmlDoc.querySelector('robot')?.getAttribute('name') || 'Robot', links: {}, joints: {} };

  Array.from(xmlDoc.getElementsByTagName('link')).forEach(link => {
    const name = link.getAttribute('name');
    const visuals = [];
    Array.from(link.querySelectorAll(':scope > visual')).forEach(visual => {
      const mesh = visual.querySelector('geometry mesh');
      if (mesh) {
        const filename = mesh.getAttribute('filename').split('/').pop();
        const origin = visual.querySelector('origin');
        const xyz = origin && origin.getAttribute('xyz') ? origin.getAttribute('xyz').split(' ').map(Number) : [0,0,0];
        const rpy = origin && origin.getAttribute('rpy') ? origin.getAttribute('rpy').split(' ').map(Number) : [0,0,0];
        const scaleAttr = mesh.getAttribute('scale');
        const scale = scaleAttr ? scaleAttr.split(' ').map(Number) : [1,1,1];
        visuals.push({ filename, xyz, rpy, scale });
      }
    });
    robot.links[name] = { name, visuals };
  });

  Array.from(xmlDoc.getElementsByTagName('joint')).forEach(j => {
    const name = j.getAttribute('name');
    const type = j.getAttribute('type');
    const parent = j.querySelector('parent')?.getAttribute('link');
    const child = j.querySelector('child')?.getAttribute('link');
    const origin = j.querySelector('origin');
    const xyz = origin && origin.getAttribute('xyz') ? origin.getAttribute('xyz').split(' ').map(Number) : [0,0,0];
    const rpy = origin && origin.getAttribute('rpy') ? origin.getAttribute('rpy').split(' ').map(Number) : [0,0,0];
    const axisElem = j.querySelector('axis');
    const axis = axisElem && axisElem.getAttribute('xyz') ? axisElem.getAttribute('xyz').split(' ').map(Number) : [1,0,0];
    const limitElem = j.querySelector('limit');
    const limitLower = limitElem && limitElem.getAttribute('lower') ? parseFloat(limitElem.getAttribute('lower')) : -Math.PI;
    const limitUpper = limitElem && limitElem.getAttribute('upper') ? parseFloat(limitElem.getAttribute('upper')) : Math.PI;

    if (child) {
        robot.joints[child] = { name, type, parent, child, xyz, rpy, axis, limitLower, limitUpper };
    }
  });

  const isChild = new Set(Object.keys(robot.joints));
  robot.rootLink = Object.keys(robot.links).find(name => !isChild.has(name)) || Object.keys(robot.links)[0];
  return robot;
}

function clampJoint(bone, jointData) {
    if (!jointData || jointData.type === 'fixed') return;
    const qTotal = bone.quaternion.clone();
    const qOrigin = new THREE.Quaternion().setFromEuler(new THREE.Euler(jointData.rpy[0], jointData.rpy[1], jointData.rpy[2], 'ZYX'));
    const qJoint = qOrigin.clone().invert().multiply(qTotal);
    const axisVec = new THREE.Vector3(...jointData.axis).normalize();
    const sinHalfTheta = qJoint.x * axisVec.x + qJoint.y * axisVec.y + qJoint.z * axisVec.z;
    const cosHalfTheta = qJoint.w;
    let angle = 2 * Math.atan2(sinHalfTheta, cosHalfTheta);
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    const clampedAngle = THREE.MathUtils.clamp(angle, jointData.limitLower, jointData.limitUpper);
    const qJointClamped = new THREE.Quaternion().setFromAxisAngle(axisVec, clampedAngle);
    bone.quaternion.copy(qOrigin.multiply(qJointClamped));
}

// ==========================================
// 1. 核心数学与图表
// ==========================================
function cubicBezier(t, p0, p1, p2, p3) { return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3; }
function solveBezier(x, x1, y1, x2, y2) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  if (x1 === y1 && x2 === y2) return x; 
  let lower = 0, upper = 1, t = x;
  for (let i = 0; i < 15; i++) {
    let currentX = cubicBezier(t, 0, x1, x2, 1);
    if (Math.abs(currentX - x) < 0.001) break;
    if (currentX < x) lower = t; else upper = t;
    t = (upper + lower) / 2;
  }
  return cubicBezier(t, 0, y1, y2, 1);
}

function evaluateFrame(frame, keyframes, robotData) {
    const state = {};
    const allBones = [robotData.rootLink, ...Object.keys(robotData.joints)];
    
    allBones.forEach(boneName => {
        const keys = keyframes[boneName];
        let defaultQ = [0,0,0,1]; let defaultP = [0,0,0];
        const jointData = robotData.joints[boneName];
        if (jointData) {
            defaultQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(jointData.rpy[0], jointData.rpy[1], jointData.rpy[2], 'ZYX')).toArray();
            defaultP = [...jointData.xyz];
        }

        if (!keys) { state[boneName] = { q: defaultQ, p: defaultP }; return; }
        if (keys[frame]) { state[boneName] = { q: keys[frame].v, p: keys[frame].p || defaultP }; return; }
        
        let pF = -1, nF = Infinity;
        for (let f in keys) {
            const numF = Number(f);
            if (numF < frame && numF > pF) pF = numF;
            if (numF > frame && numF < nF) nF = numF;
        }
        
        if (pF !== -1 && nF !== Infinity) {
            const linear_t = (frame - pF) / (nF - pF);
            const curve = keys[nF].c || [0.25, 0.25, 0.75, 0.75];
            const eased_t = solveBezier(linear_t, curve[0], curve[1], curve[2], curve[3]);
            const qPrev = new THREE.Quaternion().fromArray(keys[pF].v);
            const qNext = new THREE.Quaternion().fromArray(keys[nF].v);
            const pPrev = new THREE.Vector3().fromArray(keys[pF].p || defaultP);
            const pNext = new THREE.Vector3().fromArray(keys[nF].p || defaultP);
            state[boneName] = { q: qPrev.slerp(qNext, eased_t).toArray(), p: pPrev.lerp(pNext, eased_t).toArray() };
        } else if (pF !== -1) {
            state[boneName] = { q: keys[pF].v, p: keys[pF].p || defaultP };
        } else if (nF !== Infinity) {
            state[boneName] = { q: keys[nF].v, p: keys[nF].p || defaultP };
        } else {
            state[boneName] = { q: defaultQ, p: defaultP };
        }
    });
    return state;
}

const IconStart = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="19 20 9 12 19 4 19 20"></polygon><line x1="5" y1="19" x2="5" y2="5"></line></svg>;
const IconEnd = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 4 15 12 5 20 5 4"></polygon><line x1="19" y1="5" x2="19" y2="19"></line></svg>;
const IconPrevKey = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="15 18 9 12 15 6 15 18"></polygon><line x1="7" y1="6" x2="7" y2="18"></line></svg>;
const IconNextKey = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="9 6 15 12 9 18 9 6"></polygon><line x1="17" y1="6" x2="17" y2="18"></line></svg>;
const IconPlay = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>;
const IconPause = () => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>;

// ==========================================
// 2. 曲线编辑器
// ==========================================
const CurveEditor = ({ curve, onChange, disabled, theme }) => {
  const canvasRef = useRef(null);
  const [dragging, setDragging] = useState(0); 

  const draw = () => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); const rect = canvas.parentElement.getBoundingClientRect();
    if(rect.width === 0 || rect.height === 0) return; 
    canvas.width = rect.width; canvas.height = rect.height - 30; ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (disabled) { 
        ctx.fillStyle = theme === 'dark' ? '#555' : '#aaa'; 
        ctx.font = '12px sans-serif'; ctx.textAlign = 'center'; 
        ctx.fillText('选中第二帧以激活', canvas.width / 2, canvas.height / 2); return; 
    }
    ctx.strokeStyle = theme === 'dark' ? '#3e3e42' : '#cccccc'; ctx.lineWidth = 1; ctx.beginPath(); 
    ctx.moveTo(0, canvas.height / 2); ctx.lineTo(canvas.width, canvas.height / 2); ctx.moveTo(canvas.width / 2, 0); ctx.lineTo(canvas.width / 2, canvas.height); ctx.stroke();
    const pad = 20; const w = canvas.width - pad * 2; const h = canvas.height - pad * 2;
    const [x1, y1, x2, y2] = curve; const p1x = pad + x1 * w; const p1y = pad + (1 - y1) * h; const p2x = pad + x2 * w; const p2y = pad + (1 - y2) * h;
    
    ctx.strokeStyle = theme === 'dark' ? '#c586c0' : '#8a2be2'; 
    ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(pad, pad + h); ctx.bezierCurveTo(p1x, p1y, p2x, p2y, pad + w, pad); ctx.stroke();
    
    ctx.strokeStyle = theme === 'dark' ? '#555' : '#888'; ctx.beginPath(); ctx.moveTo(pad, pad + h); ctx.lineTo(p1x, p1y); ctx.moveTo(pad + w, pad); ctx.lineTo(p2x, p2y); ctx.stroke();
    
    const activeColor = theme === 'dark' ? '#0098ff' : '#005a9e';
    const idleColor = theme === 'dark' ? '#007acc' : '#007acc';
    ctx.fillStyle = dragging === 1 ? activeColor : idleColor; ctx.fillRect(p1x - 4, p1y - 4, 8, 8);
    ctx.fillStyle = dragging === 2 ? activeColor : idleColor; ctx.fillRect(p2x - 4, p2y - 4, 8, 8);
  };

  useEffect(draw, [curve, disabled, dragging, theme]);
  useEffect(() => { window.addEventListener('resize', draw); return () => window.removeEventListener('resize', draw); }, []);

  const handlePointerDown = (e) => {
    if (disabled) return;
    const rect = canvasRef.current.getBoundingClientRect(); const ex = e.clientX - rect.left; const ey = e.clientY - rect.top;
    const pad = 20; const w = canvasRef.current.width - pad * 2; const h = canvasRef.current.height - pad * 2;
    const [x1, y1, x2, y2] = curve; const p1x = pad + x1 * w; const p1y = pad + (1 - y1) * h; const p2x = pad + x2 * w; const p2y = pad + (1 - y2) * h;
    if (Math.hypot(ex - p1x, ey - p1y) < 15) setDragging(1); else if (Math.hypot(ex - p2x, ey - p2y) < 15) setDragging(2);
  };
  const handlePointerMove = (e) => {
    if (!dragging) return;
    const rect = canvasRef.current.getBoundingClientRect(); const pad = 20; const w = canvasRef.current.width - pad * 2; const h = canvasRef.current.height - pad * 2;
    let nx = Math.max(0, Math.min(1, (e.clientX - rect.left - pad) / w)); let ny = Math.max(0, Math.min(1, 1 - (e.clientY - rect.top - pad) / h));
    const newCurve = [...curve]; if (dragging === 1) { newCurve[0] = nx; newCurve[1] = ny; } if (dragging === 2) { newCurve[2] = nx; newCurve[3] = ny; }
    onChange(newCurve);
  };

  return (
    <div className="flex-1 flex flex-col bg-[var(--bg-main)] border-t-2 border-[var(--border)] min-h-[200px]">
      <div className="h-[30px] bg-[var(--bg-header)] px-3 py-1 text-xs font-bold border-b border-[var(--border)] flex justify-between items-center text-[var(--text-main)]">
        <span>插值曲线</span>
        {!disabled && <span className="font-normal" style={{color: theme === 'dark' ? '#c586c0' : '#8a2be2'}}>P1({curve[0].toFixed(2)}, {curve[1].toFixed(2)})</span>}
      </div>
      <div className="flex-1 w-full relative">
         <canvas ref={canvasRef} className={`absolute inset-0 block ${disabled ? 'cursor-not-allowed' : 'cursor-crosshair'}`} onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={() => setDragging(0)} onPointerLeave={() => setDragging(0)} />
      </div>
    </div>
  );
};

// ==========================================
// 3. 高性能时间轴组件 (边界安全版)
// ==========================================
const Timeline = ({ 
  robotData, boneNames, totalFrames, setTotalFrames,
  currentFrame, setCurrentFrame, keyframes, 
  selectedBones, onSelectBone, activeKey, setActiveKey, 
  selectionBox, setSelectionBox, isDraggingTimeline, setIsDraggingTimeline,
  clipboard, onCopy, onPaste, onDelete, onSelectBoneRange,
  isPlaying, setIsPlaying, onStart, onEnd, onPrevKey, onNextKey, onRequestChangeTotalFrames, theme
}) => {
  const scrollRef = useRef(null);
  const timelineContentRef = useRef(null);
  const keyframeCanvasRef = useRef(null); 
  const [rangeStart, setRangeStart] = useState(0); 
  const [rangeEnd, setRangeEnd] = useState(60);

  useEffect(() => {
    if (isPlaying && scrollRef.current) {
      const container = scrollRef.current; const targetX = currentFrame * 20; const viewWidth = container.clientWidth; const currentScroll = container.scrollLeft;
      if (targetX > currentScroll + viewWidth - 140) container.scrollTo({ left: targetX - viewWidth / 2, behavior: 'auto' });
      else if (targetX < currentScroll) container.scrollTo({ left: Math.max(0, targetX - 40), behavior: 'auto' });
    }
  }, [currentFrame, isPlaying]);

  const drawCanvas = useCallback(() => {
      const canvas = keyframeCanvasRef.current;
      const scrollEl = scrollRef.current;
      if (!canvas || !scrollEl) return;

      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      const dpr = window.devicePixelRatio || 1;
      const displayW = Math.floor(rect.width * dpr);
      const displayH = Math.floor(rect.height * dpr);
      
      if (canvas.width !== displayW || canvas.height !== displayH) {
          canvas.width = displayW; canvas.height = displayH;
      }

      const ctx = canvas.getContext('2d');
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const sL = scrollEl.scrollLeft;
      const sT = scrollEl.scrollTop;

      const startFrame = Math.max(0, Math.floor(sL / 20) - 1);
      const endFrame = Math.min(totalFrames, Math.ceil((sL + rect.width) / 20) + 1);
      const startBone = Math.max(0, Math.floor(sT / 24) - 1);
      const endBone = Math.min(boneNames.length - 1, Math.ceil((sT + rect.height) / 24) + 1);

      ctx.fillStyle = theme === 'dark' ? '#c586c0' : '#8a2be2';
      for (let bIdx = startBone; bIdx <= endBone; bIdx++) {
          const bone = boneNames[bIdx];
          const keys = keyframes[bone];
          if (!keys) continue;

          const cy = bIdx * 24 + 12 - sT;
          for (let f = startFrame; f <= endFrame; f++) {
              if (keys[f]) {
                  if (activeKey?.bone === bone && activeKey?.frame === f) continue;
                  const cx = f * 20 + 10 - sL;
                  ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, Math.PI * 2); ctx.fill();
              }
          }
      }

      if (activeKey && keyframes[activeKey.bone]?.[activeKey.frame]) {
          const bIdx = boneNames.indexOf(activeKey.bone);
          if (bIdx >= startBone && bIdx <= endBone && activeKey.frame >= startFrame && activeKey.frame <= endFrame) {
              const cx = activeKey.frame * 20 + 10 - sL;
              const cy = bIdx * 24 + 12 - sT;
              ctx.fillStyle = theme === 'dark' ? '#ffffff' : '#000000'; 
              ctx.shadowColor = theme === 'dark' ? '#ffffff' : '#000000'; 
              ctx.shadowBlur = 6;
              ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
              ctx.shadowBlur = 0;
          }
      }
      ctx.restore();
  }, [keyframes, activeKey, totalFrames, boneNames, theme]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);
  useEffect(() => {
      const scrollEl = scrollRef.current;
      if (scrollEl) {
          scrollEl.addEventListener('scroll', drawCanvas);
          window.addEventListener('resize', drawCanvas);
          return () => { scrollEl.removeEventListener('scroll', drawCanvas); window.removeEventListener('resize', drawCanvas); };
      }
  }, [drawCanvas]);

  const getCoordinatesFromEvent = (e) => {
      const rect = scrollRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
      if (mouseX < 150 || mouseY < 24) return null; 
      const sL = scrollRef.current.scrollLeft; const sT = scrollRef.current.scrollTop;
      const x = mouseX - 150 + sL; const y = mouseY - 24 + sT;
      const frame = Math.max(0, Math.min(totalFrames, Math.floor(x / 20)));
      const boneIdx = Math.max(0, Math.min(boneNames.length - 1, Math.floor(y / 24)));
      return { frame, boneIdx };
  };

  const handlePointerDown = (e) => {
      const coords = getCoordinatesFromEvent(e); if (!coords) return;
      const { frame, boneIdx } = coords;
      setIsDraggingTimeline(true); setSelectionBox({ startBone: boneIdx, endBone: boneIdx, startFrame: frame, endFrame: frame }); setCurrentFrame(frame);
      const boneName = boneNames[boneIdx];
      if (!selectedBones.includes(boneName)) onSelectBone([boneName]);
      if (keyframes[boneName] && keyframes[boneName][frame]) setActiveKey({ bone: boneName, frame }); else setActiveKey(null);
  };
  const handlePointerMove = (e) => {
      if (!isDraggingTimeline) return;
      const coords = getCoordinatesFromEvent(e); if (!coords) return;
      setSelectionBox(prev => ({ ...prev, endBone: coords.boneIdx, endFrame: coords.frame }));
  };
  const handleScrollToCurrent = () => { if (scrollRef.current) scrollRef.current.scrollTo({ left: Math.max(0, currentFrame * 20 - 50), behavior: 'smooth' }); };

  const bgGrid = `repeating-linear-gradient(to right, var(--border) 0px, transparent 1px, transparent 20px)`;

  return (
    <div className="flex-[2] flex flex-col bg-[var(--bg-panel)] overflow-hidden min-h-[250px] relative">
      <div className="bg-[var(--bg-header)] px-3 py-1 text-xs font-bold border-b border-[var(--border)] flex justify-between items-center text-[var(--text-main)] h-[30px] shrink-0">
        <span>帧操作记录</span>
        <div className="flex items-center gap-2">
          <span style={{color: 'var(--accent)'}}>当前: {currentFrame} / {totalFrames}</span>
          <button onClick={onRequestChangeTotalFrames} className="px-2 py-0.5 bg-[var(--bg-hover)] hover:bg-[var(--border)] text-[var(--text-main)] rounded text-[10px] transition-colors">⚙️ 设置</button>
        </div>
      </div>
      
      <div className="flex-1 relative overflow-hidden flex flex-col bg-[var(--bg-main)]">
         <canvas ref={keyframeCanvasRef} className="absolute pointer-events-none z-20" style={{ left: 150, top: 24, width: 'calc(100% - 168px)', height: 'calc(100% - 42px)' }} />

         <div ref={scrollRef} className="flex-1 overflow-auto relative select-none" onPointerDown={handlePointerDown} onPointerMove={handlePointerMove}>
            <div style={{ width: `${totalFrames * 20 + 150}px`, height: `${boneNames.length * 24 + 24}px` }} className="relative">
               <div className="sticky top-0 h-[24px] flex z-40 bg-[var(--bg-header)] border-b border-[var(--border)]">
                  <div className="sticky left-0 w-[150px] bg-[var(--bg-header)] z-50 border-r border-[var(--border)] flex items-center pl-2 text-xs font-bold text-[var(--text-main)]" onPointerDown={e=>e.stopPropagation()}>节点 (Links)</div>
                  <div className="flex-1 relative" style={{ backgroundImage: `repeating-linear-gradient(to right, transparent, transparent 19px, var(--border) 19px, var(--border) 20px)` }}>
                     {Array.from({length: Math.floor(totalFrames/5) + 1}).map((_, i) => (
                       <div key={i} style={{ position:'absolute', left: i*5*20, width: 20, textAlign: 'center', lineHeight: '24px' }}>
                          {i*5 % 10 === 0 ? <span className="font-bold text-[var(--text-highlight)] text-[10px]">{i*5}</span> : <span className="text-[9px] text-[var(--text-muted)]">{i*5}</span>}
                       </div>
                     ))}
                  </div>
               </div>
               <div className="relative" style={{ backgroundImage: bgGrid, backgroundPosition: '150px 0', backgroundSize: '20px 100%' }}>
                  {boneNames.map((bone, boneIdx) => {
                     const isRoot = robotData && bone === robotData.rootLink;
                     const isSelected = selectedBones.includes(bone);
                     return (
                       <div key={bone} style={{backgroundColor: isSelected ? 'var(--bg-selected)' : 'transparent'}} className={`flex h-[24px] border-b border-[var(--border)] hover:bg-[var(--bg-hover)] transition-colors`}>
                          <div className={`sticky left-0 w-[150px] z-30 p-1 font-mono text-[10px] border-r border-[var(--border)] overflow-hidden text-ellipsis whitespace-nowrap cursor-pointer ${isSelected ? 'bg-[var(--accent)] text-white' : (isRoot ? 'text-[#e55353] font-bold' : 'text-[var(--text-main)] bg-[var(--bg-panel)]')}`} onPointerDown={(e) => { e.stopPropagation(); onSelectBone([bone]); }}>
                             {bone} {isRoot ? '(Base)' : ''}
                          </div>
                       </div>
                     );
                  })}
                  <div style={{ position: 'absolute', left: 150 + currentFrame * 20, top: 0, width: 20, height: '100%', backgroundColor: 'rgba(0,122,204,0.2)', pointerEvents: 'none', zIndex: 10, borderLeft: '1px solid rgba(0,122,204,0.8)' }} />
                  {selectionBox && <div style={{ position: 'absolute', left: 150 + Math.min(selectionBox.startFrame, selectionBox.endFrame) * 20, top: Math.min(selectionBox.startBone, selectionBox.endBone) * 24, width: (Math.abs(selectionBox.endFrame - selectionBox.startFrame) + 1) * 20, height: (Math.abs(selectionBox.endBone - selectionBox.startBone) + 1) * 24, backgroundColor: 'rgba(0,152,255,0.3)', border: '1px solid #0098ff', pointerEvents: 'none', zIndex: 15 }} />}
               </div>
            </div>
         </div>
      </div>

      <div className="h-[32px] bg-[var(--bg-header)] border-t border-[var(--border)] flex items-center px-2 gap-1.5 text-xs text-[var(--text-main)] z-40 relative overflow-x-auto whitespace-nowrap shrink-0">
        <button onClick={handleScrollToCurrent} className="px-2 py-0.5 bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] text-[var(--text-btn)] rounded border border-[var(--border)] text-[11px]">定位</button>
        <div className="w-px h-3 bg-[var(--text-muted)] mx-0.5" />
        <button onClick={onCopy} className="px-2 py-0.5 bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] text-[var(--text-btn)] rounded border border-[var(--border)] text-[11px]">复制</button>
        <button onClick={onPaste} disabled={!clipboard || clipboard.length === 0} className={`px-2 py-0.5 rounded text-[11px] border border-[var(--border)] ${clipboard?.length > 0 ? 'bg-[var(--accent)] text-white' : 'bg-[var(--bg-btn)] text-[var(--text-muted)] cursor-not-allowed'}`}>粘贴</button>
        <button onClick={onDelete} className="px-2 py-0.5 bg-[#d32f2f] hover:bg-[#b71c1c] text-white rounded text-[11px]">删除</button>
        <div className="flex items-center gap-1 ml-auto">
          <input type="number" min={0} max={totalFrames} value={rangeStart} onChange={(e) => setRangeStart(Number(e.target.value))} className="w-10 bg-[var(--bg-main)] border border-[var(--border)] rounded text-[var(--text-main)] text-center text-[10px] outline-none p-0.5" />
          <span className="text-[10px] text-[var(--text-muted)]">-</span>
          <input type="number" min={0} max={totalFrames} value={rangeEnd} onChange={(e) => setRangeEnd(Number(e.target.value))} className="w-10 bg-[var(--bg-main)] border border-[var(--border)] rounded text-[var(--text-main)] text-center text-[10px] outline-none p-0.5" />
          <button onClick={() => onSelectBoneRange(rangeStart, rangeEnd)} className="px-2 py-0.5 bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] text-[var(--text-btn)] border border-[var(--border)] rounded text-[11px]">框选</button>
        </div>
      </div>
      <div className="h-[40px] bg-[var(--bg-panel)] border-t border-[var(--border)] flex items-center justify-center gap-2 text-[var(--text-main)] z-40 relative shrink-0">
        <button onClick={onStart} className="p-1.5 hover:bg-[var(--bg-hover)] rounded"><IconStart /></button>
        <button onClick={onPrevKey} className="p-1.5 hover:bg-[var(--bg-hover)] rounded"><IconPrevKey /></button>
        <button onClick={() => setIsPlaying(!isPlaying)} className={`p-1.5 rounded ${isPlaying ? 'bg-[var(--accent)] text-white' : 'hover:bg-[var(--bg-hover)]'}`}>
          {isPlaying ? <IconPause /> : <IconPlay />}
        </button>
        <button onClick={onNextKey} className="p-1.5 hover:bg-[var(--bg-hover)] rounded"><IconNextKey /></button>
        <button onClick={onEnd} className="p-1.5 hover:bg-[var(--bg-hover)] rounded"><IconEnd /></button>
      </div>
    </div>
  );
};

// ==========================================
// 4. 支持真实 URDF 物理渲染的 3D 视图引擎
// ==========================================
const Viewport3D = forwardRef(({ mode, selType, space, ikMode, currentFrame, keyframes, isPlaying, selectedBones, onSelectBone, robotData, fileMap, theme }, ref) => {
  const containerRef = useRef(null);
  const boxRef = useRef(null); 

  const stateRef = useRef({ 
    mode, selType, space, ikMode, isPlaying, selectedBones,
    bonesMap: {}, transformControl: null, ikTargetMesh: null, camera: null, orbit: null, scene: null, grid: null
  });

  useEffect(() => { 
      stateRef.current.mode = mode; stateRef.current.selType = selType; stateRef.current.selectedBones = selectedBones; 
      if (stateRef.current.orbit) stateRef.current.orbit.mouseButtons.LEFT = mode === 'select' ? null : THREE.MOUSE.ROTATE;
      if (stateRef.current.transformControl) {
          if (mode === 'select') stateRef.current.transformControl.detach();
          else if (mode === 'rotate' || mode === 'move') {
               const activeBoneName = selectedBones[selectedBones.length - 1];
               const activeBone = stateRef.current.bonesMap[activeBoneName];
               if (activeBone && activeBoneName === robotData.rootLink) {
                   stateRef.current.transformControl.setMode(mode === 'rotate' ? 'rotate' : 'translate');
                   stateRef.current.transformControl.showX = stateRef.current.transformControl.showY = stateRef.current.transformControl.showZ = true;
                   stateRef.current.transformControl.attach(activeBone);
               } else if (mode === 'rotate' && activeBone) {
                   stateRef.current.transformControl.setMode('rotate');
                   const jointData = activeBone.userData.jointData;
                   if (jointData && jointData.axis) {
                       stateRef.current.transformControl.showX = Math.abs(jointData.axis[0]) > 0.5;
                       stateRef.current.transformControl.showY = Math.abs(jointData.axis[1]) > 0.5;
                       stateRef.current.transformControl.showZ = Math.abs(jointData.axis[2]) > 0.5;
                   } else {
                       stateRef.current.transformControl.showX = stateRef.current.transformControl.showY = stateRef.current.transformControl.showZ = true;
                   }
                   stateRef.current.transformControl.attach(activeBone);
               } else if (mode === 'move' && stateRef.current.ikTargetMesh && activeBone) {
                   stateRef.current.transformControl.setMode('translate');
                   stateRef.current.transformControl.showX = stateRef.current.transformControl.showY = stateRef.current.transformControl.showZ = true;
                   stateRef.current.transformControl.attach(stateRef.current.ikTargetMesh);
               } else stateRef.current.transformControl.detach();
          }
      }
  }, [mode, selType, selectedBones]);

  useEffect(() => { stateRef.current.isPlaying = isPlaying; }, [isPlaying]);
  useEffect(() => { stateRef.current.space = space; if (stateRef.current.transformControl) stateRef.current.transformControl.setSpace(space); }, [space]);
  useEffect(() => { stateRef.current.ikMode = ikMode; }, [ikMode]);

  // 动态切换明暗场景
  useEffect(() => {
      if (stateRef.current.scene && stateRef.current.grid) {
          stateRef.current.scene.background = new THREE.Color(theme === 'dark' ? 0x222222 : 0xf0f0f0);
          stateRef.current.grid.material.color.setHex(theme === 'dark' ? 0x444444 : 0xcccccc);
      }
  }, [theme]);

  // 🔴 智能反馈：提供将被修改的 IK 骨骼链集合给外部注册
  useImperativeHandle(ref, () => ({
    getAllBoneRotations: () => {
      const res = {};
      Object.entries(stateRef.current.bonesMap).forEach(([name, bone]) => { res[name] = { v: bone.quaternion.toArray(), p: bone.position.toArray() }; });
      return res;
    },
    resetSelectedBones: () => {
      stateRef.current.selectedBones.forEach(boneName => {
         const bone = stateRef.current.bonesMap[boneName];
         if(bone) {
             const baseR = bone.userData.baseRotation;
             if(baseR) bone.quaternion.setFromEuler(new THREE.Euler().setFromVector3(baseR));
             if(bone.userData.basePosition) bone.position.copy(bone.userData.basePosition);
         }
      });
    },
    getAffectedBones: () => {
        if (stateRef.current.mode === 'move') {
            const activeBoneName = stateRef.current.selectedBones[stateRef.current.selectedBones.length - 1];
            const activeBone = stateRef.current.bonesMap[activeBoneName];
            if (!activeBone || activeBoneName === robotData.rootLink) return stateRef.current.selectedBones;

            const affected = new Set(stateRef.current.selectedBones);
            let current = activeBone.parent;
            
            // 🔴 智能判定 IK 影响的节点：一旦碰到未被选中的父节点，立即切断！完美解决框选拉扯问题
            while (current && current.type !== 'Group') {
                if (current.isBone && current.name && current.name !== robotData.rootLink) {
                    const jointData = current.userData.jointData;
                    if (jointData && jointData.type !== 'fixed') {
                        if (stateRef.current.ikMode !== 'global' && !stateRef.current.selectedBones.includes(current.name)) {
                            break; // 在局部模式下，遇到未选中的父节点立即终止链的追溯
                        }
                        affected.add(current.name);
                    }
                }
                current = current.parent;
            }
            return Array.from(affected);
        }
        return stateRef.current.selectedBones;
    }
  }));

  useEffect(() => {
    const { bonesMap } = stateRef.current;
    if (!bonesMap || Object.keys(bonesMap).length === 0) return;
    const state = evaluateFrame(currentFrame, keyframes, robotData);
    Object.keys(state).forEach(boneName => {
        const bone = bonesMap[boneName];
        if (bone) {
            bone.quaternion.fromArray(state[boneName].q);
            bone.position.fromArray(state[boneName].p);
        }
    });
  }, [currentFrame, keyframes]);

  useEffect(() => {
    if (!containerRef.current || !robotData) return;
    const container = containerRef.current;
    Array.from(container.children).forEach(c => { if(c.tagName === 'CANVAS') container.removeChild(c) });

    const scene = new THREE.Scene(); 
    scene.background = new THREE.Color(theme === 'dark' ? 0x222222 : 0xf0f0f0);
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 1.5, 3); 
    stateRef.current.camera = camera;
    stateRef.current.scene = scene;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0); dirLight.position.set(5, 10, 5); scene.add(dirLight);
    
    const grid = new THREE.GridHelper(10, 20, 0x444444, theme === 'dark' ? 0x444444 : 0xcccccc);
    scene.add(grid);
    stateRef.current.grid = grid;

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(0, 0.5, 0); orbit.update();
    stateRef.current.orbit = orbit;

    const transformControl = new TransformControls(camera, renderer.domElement);
    if (!transformControl.isObject3D) {
      transformControl.isObject3D = true; const dummy = new THREE.Object3D();
      for (const key in dummy) if (transformControl[key] === undefined) transformControl[key] = dummy[key];
      Object.getOwnPropertyNames(THREE.Object3D.prototype).forEach(method => { if (method !== 'constructor' && typeof transformControl[method] !== 'function') transformControl[method] = THREE.Object3D.prototype[method]; });
      Object.getOwnPropertyNames(THREE.EventDispatcher.prototype).forEach(method => { if (method !== 'constructor' && typeof transformControl[method] !== 'function') transformControl[method] = THREE.EventDispatcher.prototype[method]; });
      if (!transformControl.children) transformControl.children = [];
      ['_root', '_gizmo', '_plane'].forEach(key => { if (transformControl[key]) { transformControl[key].parent = transformControl; if (!transformControl.children.includes(transformControl[key])) transformControl.children.push(transformControl[key]); } });
    }
    if (!transformControl._worldScale) transformControl._worldScale = new THREE.Vector3();
    if (typeof transformControl.removeFromParent !== 'function') transformControl.removeFromParent = function() { if (this.parent) this.parent.remove(this); };

    transformControl.addEventListener('change', () => {
        if (stateRef.current.mode === 'rotate' && transformControl.object) {
            const bone = transformControl.object;
            const jointData = bone.userData.jointData;
            if (jointData) clampJoint(bone, jointData);
        }
    });
    transformControl.addEventListener('dragging-changed', (event) => { orbit.enabled = !event.value; });
    scene.add(transformControl);

    const bonesMap = {}; const bonesArray = []; const interactables = [];
    Object.keys(robotData.links).forEach(linkName => {
        bonesMap[linkName] = new THREE.Bone(); bonesMap[linkName].name = linkName; bonesArray.push(bonesMap[linkName]);
    });

    const rootTransformGroup = new THREE.Group(); scene.add(rootTransformGroup);
    rootTransformGroup.rotation.x = -Math.PI / 2; rootTransformGroup.updateMatrixWorld(true);

    if (bonesMap[robotData.rootLink]) {
        rootTransformGroup.add(bonesMap[robotData.rootLink]);
        bonesMap[robotData.rootLink].userData.basePosition = bonesMap[robotData.rootLink].position.clone();
        bonesMap[robotData.rootLink].userData.baseRotation = bonesMap[robotData.rootLink].rotation.clone();
        
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.06), new THREE.MeshBasicMaterial({ color: 0xffaa00, transparent: true, opacity: 0.8, depthTest: false }));
        sphere.renderOrder = 1; sphere.userData = { isJoint: true, targetBone: bonesMap[robotData.rootLink] };
        bonesMap[robotData.rootLink].add(sphere); interactables.push(sphere);
    }

    let latestDynamicBone = null;
    Object.values(robotData.joints).forEach(joint => {
        const childBone = bonesMap[joint.child];
        const parentBone = bonesMap[joint.parent] || rootTransformGroup;

        if (childBone) {
            childBone.position.set(...joint.xyz);
            childBone.quaternion.setFromEuler(new THREE.Euler(joint.rpy[0], joint.rpy[1], joint.rpy[2], 'ZYX'));
            childBone.userData.jointData = joint;
            childBone.userData.basePosition = childBone.position.clone();
            childBone.userData.baseRotation = childBone.rotation.clone(); 
            parentBone.add(childBone);

            if (joint.type !== 'fixed') {
                latestDynamicBone = childBone;
                const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.04), new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.6, depthTest: false }));
                sphere.renderOrder = 1; sphere.userData = { isJoint: true, targetBone: childBone };
                childBone.add(sphere); interactables.push(sphere);
            }
        }
    });

    stateRef.current.bonesMap = bonesMap;

    const stlLoader = new STLLoader();
    const defaultMaterial = new THREE.MeshPhongMaterial({ color: 0x999999, specular: 0x111111, shininess: 50 });
    Object.values(robotData.links).forEach(link => {
        const bone = bonesMap[link.name]; if (!bone) return;
        let hasVisual = false;
        link.visuals.forEach(visual => {
            const fileUrl = fileMap[visual.filename] || Object.entries(fileMap).find(([p]) => p.endsWith(visual.filename))?.[1];
            if (fileUrl) {
                hasVisual = true;
                stlLoader.load(fileUrl, (geometry) => {
                    const visualMesh = new THREE.Mesh(geometry, defaultMaterial);
                    visualMesh.position.set(...visual.xyz);
                    visualMesh.quaternion.setFromEuler(new THREE.Euler(visual.rpy[0], visual.rpy[1], visual.rpy[2], 'ZYX'));
                    visualMesh.scale.set(...visual.scale);
                    bone.add(visualMesh);
                });
            }
        });
        if (!hasVisual && link.name !== robotData.rootLink && bone.userData.jointData?.type !== 'fixed') {
             const length = 0.15; const cylGeo = new THREE.CylinderGeometry(0.015, 0.015, length, 8); cylGeo.translate(0, length/2, 0);
             bone.add(new THREE.Mesh(cylGeo, new THREE.MeshPhongMaterial({ color: 0x777777, wireframe: true })));
        }
    });

    const targetBone = new THREE.Bone();
    targetBone.matrixWorld = new THREE.Matrix4();
    scene.add(targetBone);
    bonesArray.push(targetBone);

    const skeleton = new THREE.Skeleton(bonesArray);
    const hostGeo = new THREE.BufferGeometry();
    const indices = new Uint16Array(bonesArray.length * 4); const weights = new Float32Array(bonesArray.length * 4);
    indices[0] = 0; weights[0] = 1; 
    hostGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0,0,0]), 3));
    hostGeo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(indices, 4));
    hostGeo.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
    const hostMesh = new THREE.SkinnedMesh(hostGeo, new THREE.MeshBasicMaterial({visible:false}));
    hostMesh.add(rootTransformGroup);
    hostMesh.bind(skeleton);
    scene.add(hostMesh);

    const ikTargetMesh = new THREE.Mesh(new THREE.SphereGeometry(0.05), new THREE.MeshBasicMaterial({ color: 0xff3333, depthTest: false }));
    ikTargetMesh.renderOrder = 2; ikTargetMesh.userData = { isIKTarget: true }; scene.add(ikTargetMesh);
    interactables.push(ikTargetMesh);

    let ikSolver = new CCDIKSolver(hostMesh, []);

    const raycaster = new THREE.Raycaster(); const mouse = new THREE.Vector2();
    let isBoxSelecting = false; let startMouse = { x: 0, y: 0 };

    const onPointerDownSelect = (e) => {
      if (e.button !== 0 || stateRef.current.mode !== 'select') return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ex = e.clientX - rect.left; const ey = e.clientY - rect.top;
      if (stateRef.current.selType === 'box') {
          isBoxSelecting = true; startMouse = { x: ex, y: ey };
          if (boxRef.current) {
              boxRef.current.style.display = 'block'; boxRef.current.style.left = `${ex}px`; boxRef.current.style.top = `${ey}px`;
              boxRef.current.style.width = '0px'; boxRef.current.style.height = '0px';
          }
      } else if (stateRef.current.selType === 'free') {
          mouse.x = (ex / rect.width) * 2 - 1; mouse.y = -(ey / rect.height) * 2 + 1;
          raycaster.setFromCamera(mouse, camera);
          const intersects = raycaster.intersectObjects(interactables);
          if (intersects.length > 0 && intersects[0].object.userData.isJoint) {
             const bName = intersects[0].object.userData.targetBone.name;
             const prev = stateRef.current.selectedBones;
             onSelectBone(prev.includes(bName) ? prev.filter(b => b !== bName) : [...prev, bName]);
          }
      }
    }

    const onPointerDown = (e) => {
      if (e.button !== 0 || stateRef.current.mode === 'select') return; 
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1; mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactables);
      if (intersects.length > 0) {
        const hit = intersects[0].object;
        if (stateRef.current.mode === 'rotate' && hit.userData.isJoint) {
            const joint = hit.userData.targetBone;
            transformControl.setMode(joint.name === robotData.rootLink ? 'translate' : 'rotate'); 
            transformControl.setSpace('local'); 
            const jointData = joint.userData.jointData;
            if (jointData && jointData.axis) {
                transformControl.showX = Math.abs(jointData.axis[0]) > 0.5;
                transformControl.showY = Math.abs(jointData.axis[1]) > 0.5;
                transformControl.showZ = Math.abs(jointData.axis[2]) > 0.5;
            } else { transformControl.showX = transformControl.showY = transformControl.showZ = true; }
            transformControl.attach(joint);
            onSelectBone([joint.name]);
        } else if (stateRef.current.mode === 'move' && hit.userData.isIKTarget) {
            transformControl.setMode('translate'); 
            transformControl.showX = transformControl.showY = transformControl.showZ = true;
            transformControl.attach(ikTargetMesh);
        }
      } else if (!transformControl.dragging) {
          if (stateRef.current.mode === 'rotate') transformControl.detach();
      }
    };

    const onPointerMove = (e) => {
       if (isBoxSelecting && boxRef.current) {
          const rect = renderer.domElement.getBoundingClientRect();
          const ex = e.clientX - rect.left; const ey = e.clientY - rect.top;
          boxRef.current.style.left = `${Math.min(ex, startMouse.x)}px`; boxRef.current.style.top = `${Math.min(ey, startMouse.y)}px`;
          boxRef.current.style.width = `${Math.abs(ex - startMouse.x)}px`; boxRef.current.style.height = `${Math.abs(ey - startMouse.y)}px`;
       }
    };

    const onPointerUp = (e) => {
       if (isBoxSelecting) {
           isBoxSelecting = false; if (boxRef.current) boxRef.current.style.display = 'none';
           const rect = renderer.domElement.getBoundingClientRect();
           const endX = e.clientX - rect.left; const endY = e.clientY - rect.top;
           if (Math.abs(endX - startMouse.x) < 5 && Math.abs(endY - startMouse.y) < 5) { onSelectBone([]); return; }
           const minX = Math.min(startMouse.x, endX) / rect.width * 2 - 1; const maxX = Math.max(startMouse.x, endX) / rect.width * 2 - 1;
           const maxY = -(Math.min(startMouse.y, endY) / rect.height) * 2 + 1; const minY = -(Math.max(startMouse.y, endY) / rect.height) * 2 + 1;
           const newlySelected = [];
           Object.entries(stateRef.current.bonesMap).forEach(([name, bone]) => {
               if(bone.name === robotData.rootLink || (bone.userData.jointData && bone.userData.jointData.type !== 'fixed')) {
                   const screenPos = bone.getWorldPosition(new THREE.Vector3()).project(camera);
                   if (screenPos.x >= minX && screenPos.x <= maxX && screenPos.y >= minY && screenPos.y <= maxY && screenPos.z >= -1 && screenPos.z <= 1) newlySelected.push(name);
               }
           });
           if (newlySelected.length > 0) onSelectBone(newlySelected); else onSelectBone([]);
       }
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerdown', onPointerDownSelect);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);

    let reqId;
    const animate = () => {
      reqId = requestAnimationFrame(animate);

      interactables.forEach(obj => {
         if (obj.userData.isJoint) {
             const isSel = stateRef.current.selectedBones.includes(obj.userData.targetBone.name);
             obj.material.color.setHex(isSel ? (obj.userData.targetBone.name === robotData.rootLink ? 0xffcc00 : 0x00ff00) : (obj.userData.targetBone.name === robotData.rootLink ? 0x886600 : 0x005500)); 
             obj.scale.setScalar(isSel ? 1.5 : 1.0);
         }
      });

      const activeBoneName = stateRef.current.selectedBones[stateRef.current.selectedBones.length - 1];
      const activeBone = stateRef.current.bonesMap[activeBoneName];

      if (stateRef.current.mode === 'move' && !stateRef.current.isPlaying && activeBone && activeBoneName !== robotData.rootLink) {
        ikTargetMesh.visible = true;
        const targetBoneIndex = bonesArray.indexOf(targetBone);
        const effectorIndex = bonesArray.indexOf(activeBone);
        
        if (effectorIndex > 0) {
           if (transformControl.dragging && transformControl.object === ikTargetMesh) {
               const links = [];
               let current = activeBone.parent;
               
               // 🔴 完美断路器：如果是局部模式，必须被选中的父节点才加入IK连结！否则切断。
               while(current && current.type !== 'Group') {
                   if (current.isBone && current.name && current.name !== robotData.rootLink) {
                       const jointData = current.userData.jointData;
                       if (jointData && jointData.type !== 'fixed') {
                           if (stateRef.current.ikMode !== 'global' && !stateRef.current.selectedBones.includes(current.name)) {
                               break; 
                           }
                           const idx = bonesArray.indexOf(current);
                           if (idx > -1) {
                               links.push({ index: idx }); 
                           }
                       }
                   }
                   current = current.parent;
               }
               
               if(links.length > 0) {
                   ikSolver = new CCDIKSolver(hostMesh, [{ target: targetBoneIndex, effector: effectorIndex, links, iteration: 1 }]);
                   targetBone.position.copy(ikTargetMesh.position);
                   targetBone.updateMatrixWorld(true);
                   
                   for(let i=0; i<3; i++) {
                       ikSolver.update();
                       links.forEach(l => clampJoint(bonesArray[l.index], bonesArray[l.index].userData.jointData));
                       hostMesh.updateMatrixWorld(true);
                   }
               }
           } else {
               activeBone.getWorldPosition(ikTargetMesh.position);
           }
        }
      } else {
        ikTargetMesh.visible = false;
        if (activeBone) activeBone.getWorldPosition(ikTargetMesh.position);
        else latestDynamicBone?.getWorldPosition(ikTargetMesh.position);
        if (stateRef.current.mode !== 'move' && transformControl.object === ikTargetMesh) transformControl.detach();
      }

      if (!transformControl.dragging && !stateRef.current.isPlaying) {
         Object.entries(stateRef.current.bonesMap).forEach(([name, bone]) => {
            const jointData = bone.userData.jointData;
            if(jointData) clampJoint(bone, jointData);
         });
      }

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix(); renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', onResize);

    stateRef.current.transformControl = transformControl;
    stateRef.current.ikTargetMesh = ikTargetMesh;

    return () => {
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerdown', onPointerDownSelect);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(reqId);
      transformControl.detach(); transformControl.dispose();
      orbit.dispose(); renderer.dispose(); scene.clear(); 
      Array.from(container.children).forEach(c => { if(c.tagName === 'CANVAS') container.removeChild(c) });
    };
  }, [robotData, fileMap]); 

  return (
      <div className={`flex-1 w-full h-full relative ${mode === 'select' ? 'cursor-crosshair' : 'cursor-default'} outline-none`}>
          <div ref={containerRef} className="absolute inset-0" />
          <div ref={boxRef} className="absolute border border-[#0098ff] bg-[rgba(0,152,255,0.2)] pointer-events-none hidden z-10" />
      </div>
  );
});

// ==========================================
// 5. 主应用与状态统筹
// ==========================================
export default function App() {
  const [theme, setTheme] = useState('dark'); // 🌞/🌙 主题切换

  const [mode, setMode] = useState('rotate'); 
  const [selType, setSelType] = useState('free'); 
  const [space, setSpace] = useState('local'); 
  const [ikMode, setIkMode] = useState('selected'); 
  
  const [totalFrames, setTotalFrames] = useState(100);
  const [isFramesModalOpen, setIsFramesModalOpen] = useState(false);
  const [tempFramesInput, setTempFramesInput] = useState(100);

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const [activeKey, setActiveKey] = useState(null); 
  const [keyframes, setKeyframes] = useState({}); 

  const [timelineSelection, setTimelineSelection] = useState(null); 
  const [isDraggingTimeline, setIsDraggingTimeline] = useState(false);
  const [clipboard, setClipboard] = useState(null);

  const [robotData, setRobotData] = useState(null);
  const [activeJoints, setActiveJoints] = useState([]);
  const [selectedBones, setSelectedBones] = useState([]);
  const [fileMap, setFileMap] = useState({}); 
  const viewportRef = useRef(null);

  const [availableURDFs, setAvailableURDFs] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [isURDFModalOpen, setIsURDFModalOpen] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  // 动态主题 CSS 变量注入
  const themeStyles = theme === 'dark' ? {
      '--bg-main': '#1e1e1e', '--bg-panel': '#252526', '--bg-header': '#333333', 
      '--border': '#3e3e42', '--text-main': '#cccccc', '--text-muted': '#888888', '--text-highlight': '#ffffff',
      '--bg-hover': '#303236', '--bg-selected': '#3a3d41', '--accent': '#007acc',
      '--bg-btn': '#444444', '--bg-btn-hover': '#555555', '--text-btn': '#ffffff'
  } : {
      '--bg-main': '#f0f0f0', '--bg-panel': '#ffffff', '--bg-header': '#e8e8e8', 
      '--border': '#cccccc', '--text-main': '#333333', '--text-muted': '#666666', '--text-highlight': '#000000',
      '--bg-hover': '#f5f5f5', '--bg-selected': '#d0d0d0', '--accent': '#005a9e',
      '--bg-btn': '#ffffff', '--bg-btn-hover': '#f0f0f0', '--text-btn': '#333333'
  };

  useEffect(() => {
      const mockRobotData = parseURDF(MOCK_URDF_XML);
      const joints = Object.values(mockRobotData.joints).filter(j => j.type !== 'fixed').map(j => j.child);
      const allTimelineBones = [mockRobotData.rootLink, ...joints];
      setActiveJoints(allTimelineBones);
      setSelectedBones([allTimelineBones[0]]);
      setRobotData(mockRobotData);
  }, []);

  const handleFolderUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;
    const urdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.urdf'));
    if (urdfFiles.length === 0) { alert("没有找到 .urdf 文件！"); return; }
    if (urdfFiles.length === 1) loadSelectedURDF(urdfFiles[0], files);
    else { setAvailableURDFs(urdfFiles); setPendingFiles(files); setIsURDFModalOpen(true); }
    event.target.value = '';
  };

  const loadSelectedURDF = async (urdfFile, allFiles) => {
    setIsURDFModalOpen(false);
    const newFileMap = {};
    allFiles.forEach(f => newFileMap[f.webkitRelativePath || f.name] = URL.createObjectURL(f));
    try {
        const urdfText = await urdfFile.text();
        const newRobotData = parseURDF(urdfText);
        const joints = Object.values(newRobotData.joints).filter(j => j.type !== 'fixed').map(j => j.child);
        const allTimelineBones = [newRobotData.rootLink, ...joints]; 
        setFileMap(newFileMap); setRobotData(newRobotData); setActiveJoints(allTimelineBones);
        if (allTimelineBones.length > 0) setSelectedBones([allTimelineBones[0]]);
        setKeyframes({}); setCurrentFrame(0);
    } catch (error) { alert("URDF 解析失败！"); }
  };

  useEffect(() => { if (currentFrame > totalFrames) setCurrentFrame(totalFrames); }, [totalFrames]);
  useEffect(() => {
    const handleGlobalPointerUp = () => setIsDraggingTimeline(false);
    window.addEventListener('pointerup', handleGlobalPointerUp);
    return () => window.removeEventListener('pointerup', handleGlobalPointerUp);
  }, []);
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => setCurrentFrame(prev => (prev >= totalFrames ? 0 : prev + 1)), 1000 / 30);
    return () => clearInterval(timer);
  }, [isPlaying, totalFrames]);

  const selectAll = () => setSelectedBones([...activeJoints]);

  const handleRegisterKeyframe = () => {
    if (!viewportRef.current || selectedBones.length === 0) return;
    const currentStates = viewportRef.current.getAllBoneRotations();
    const bonesToRegister = viewportRef.current.getAffectedBones();
    
    setKeyframes(prev => {
      const nextDict = { ...prev };
      bonesToRegister.forEach(bone => {
        if (!nextDict[bone]) nextDict[bone] = {};
        const existingCurve = nextDict[bone][currentFrame]?.c || [0.25, 0.25, 0.75, 0.75]; 
        nextDict[bone] = { ...nextDict[bone], [currentFrame]: { v: currentStates[bone].v, p: currentStates[bone].p, c: existingCurve } };
      });
      return nextDict;
    });
  };

  const handleReset = () => { if (viewportRef.current) viewportRef.current.resetSelectedBones(); };

  const getSelectionBounds = () => {
    if (!timelineSelection) return null;
    return {
      minBone: Math.min(timelineSelection.startBone, timelineSelection.endBone), maxBone: Math.max(timelineSelection.startBone, timelineSelection.endBone),
      minFrame: Math.min(timelineSelection.startFrame, timelineSelection.endFrame), maxFrame: Math.max(timelineSelection.startFrame, timelineSelection.endFrame),
    };
  };

  const handleCopyFrames = () => {
    const bounds = getSelectionBounds(); if (!bounds) return;
    const newClipboard = [];
    for (let b = bounds.minBone; b <= bounds.maxBone; b++) {
      const boneName = activeJoints[b];
      for (let f = bounds.minFrame; f <= bounds.maxFrame; f++) {
        if (keyframes[boneName] && keyframes[boneName][f]) {
          newClipboard.push({ boneOffset: b - bounds.minBone, frameOffset: f - bounds.minFrame, data: JSON.parse(JSON.stringify(keyframes[boneName][f])) });
        }
      }
    }
    setClipboard(newClipboard);
  };

  const handlePasteFrames = () => {
    if (!clipboard || clipboard.length === 0) return;
    const bounds = getSelectionBounds();
    const targetBoneIdx = bounds ? bounds.minBone : Math.max(0, activeJoints.indexOf(selectedBones[0]));
    setKeyframes(prev => {
      const nextDict = { ...prev };
      clipboard.forEach(item => {
        const bIdx = targetBoneIdx + item.boneOffset; const f = currentFrame + item.frameOffset;
        if (bIdx >= 0 && bIdx < activeJoints.length && f <= totalFrames) {
          const bName = activeJoints[bIdx];
          if (!nextDict[bName]) nextDict[bName] = {};
          nextDict[bName][f] = JSON.parse(JSON.stringify(item.data));
        }
      });
      return nextDict;
    });
  };

  const handleDeleteSelectedFrames = () => {
    const bounds = getSelectionBounds(); if (!bounds) return;
    setKeyframes(prev => {
      const nextDict = { ...prev };
      for (let b = bounds.minBone; b <= bounds.maxBone; b++) {
        const boneName = activeJoints[b];
        if (nextDict[boneName]) {
          const newBoneKeys = { ...nextDict[boneName] };
          for (let f = bounds.minFrame; f <= bounds.maxFrame; f++) delete newBoneKeys[f];
          nextDict[boneName] = newBoneKeys;
        }
      }
      return nextDict;
    });
  };

  const handleSelectBoneRange = (startFrame, endFrame) => {
    if (selectedBones.length === 0) return;
    const indices = selectedBones.map(b => activeJoints.indexOf(b));
    const safeStart = Math.min(startFrame, endFrame); const safeEnd = Math.max(startFrame, endFrame);
    setTimelineSelection({ startBone: Math.min(...indices), endBone: Math.max(...indices), startFrame: Math.max(0, safeStart), endFrame: Math.min(totalFrames, safeEnd) });
  };

  const handleExportCSV = () => {
    if (!robotData) return;
    setIsImporting(true);
    setTimeout(() => {
        const csvHeaders = ["x", "y", "z", "qx", "qy", "qz", "qw"];
        const jointLinkNames = [];
        Object.values(robotData.joints).forEach(j => {
            if (j.type !== 'fixed') {
                jointLinkNames.push(j.child);
                csvHeaders.push(j.name);
            }
        });
        
        let csvContent = csvHeaders.join(",") + "\n";
        for (let f = 0; f <= totalFrames; f++) {
          const state = evaluateFrame(f, keyframes, robotData);
          const rootState = state[robotData.rootLink] || { q: [0,0,0,1], p: [0,0,0] };
          let row = `${rootState.p.join(',')},${rootState.q.join(',')}`;
          
          jointLinkNames.forEach(linkName => {
             const qArray = state[linkName]?.q || [0,0,0,1];
             const qTotal = new THREE.Quaternion().fromArray(qArray);
             const jointData = robotData.joints[linkName];
             
             const qOrigin = new THREE.Quaternion().setFromEuler(new THREE.Euler(jointData.rpy[0], jointData.rpy[1], jointData.rpy[2], 'ZYX'));
             const qJoint = qOrigin.clone().invert().multiply(qTotal);
             const axisVec = new THREE.Vector3(...jointData.axis).normalize();
             
             const sinHalfTheta = qJoint.x * axisVec.x + qJoint.y * axisVec.y + qJoint.z * axisVec.z;
             const cosHalfTheta = qJoint.w;
             let angle = 2 * Math.atan2(sinHalfTheta, cosHalfTheta);
             while (angle > Math.PI) angle -= 2 * Math.PI;
             while (angle < -Math.PI) angle += 2 * Math.PI;

             row += `,${angle.toFixed(6)}`;
          });
          csvContent += row + "\n";
        }
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = 'trajectory.csv'; a.click();
        setIsImporting(false);
    }, 50);
  };

  const handleImportCSV = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setIsImporting(true);
    setTimeout(async () => {
        try {
            const text = await file.text();
            const lines = text.trim().split('\n');
            if (lines.length < 2) throw new Error("CSV格式错误或文件过短");
            
            const headers = lines[0].split(',').map(s => s.trim());
            let startIndex = 1;
            if (!isNaN(parseFloat(headers[0]))) startIndex = 0; 
            
            const dataLines = lines.slice(startIndex).filter(l => l.trim().length > 0);
            const newTotalFrames = dataLines.length - 1; 
            setTotalFrames(Math.max(10, newTotalFrames));
            
            const newKeyframes = {};
            newKeyframes[robotData.rootLink] = {};

            const orderedJointLinks = [];
            Object.values(robotData.joints).forEach(j => {
                if (j.type !== 'fixed') orderedJointLinks.push(j.child);
            });

            const jointCache = orderedJointLinks.map(linkName => {
                const jointData = robotData.joints[linkName];
                const qOrigin = new THREE.Quaternion().setFromEuler(new THREE.Euler(jointData.rpy[0], jointData.rpy[1], jointData.rpy[2], 'ZYX'));
                const axisVec = new THREE.Vector3(...jointData.axis).normalize();
                newKeyframes[linkName] = {};
                return { linkName, qOrigin, axisVec, xyz: [...jointData.xyz] };
            });
            
            for (let i = 0; i < dataLines.length; i++) {
              const frame = i;
              const values = dataLines[i].split(',').map(Number);
              if (values.length < 7) continue;
              
              const [px, py, pz, qx, qy, qz, qw] = values;
              newKeyframes[robotData.rootLink][frame] = { v: [qx, qy, qz, qw], p: [px, py, pz], c: [0.25, 0.25, 0.75, 0.75] };
              
              jointCache.forEach((cache, idx) => {
                 const angle = values[7 + idx];
                 if (isNaN(angle)) return;
                 const qJoint = new THREE.Quaternion().setFromAxisAngle(cache.axisVec, angle);
                 const qTotal = cache.qOrigin.clone().multiply(qJoint);
                 
                 newKeyframes[cache.linkName][frame] = { 
                     v: [qTotal.x, qTotal.y, qTotal.z, qTotal.w], 
                     p: cache.xyz, 
                     c: [0.25, 0.25, 0.75, 0.75] 
                 };
              });
            }
            setKeyframes(newKeyframes);
            setCurrentFrame(0);
        } catch (e) {
            console.error(e);
            alert("导入失败: " + e.message);
        } finally {
            setIsImporting(false);
            event.target.value = '';
        }
    }, 50);
  };

  const currentActiveCurve = activeKey ? (keyframes[activeKey.bone]?.[activeKey.frame]?.c || [0.25, 0.25, 0.75, 0.75]) : [0.25, 0.25, 0.75, 0.75];
  const getAllKeyframeNums = () => {
    const frames = new Set();
    Object.values(keyframes).forEach(boneKeys => { Object.keys(boneKeys).forEach(f => frames.add(Number(f))); });
    return Array.from(frames).sort((a,b) => a-b);
  };

  return (
    <div style={themeStyles} className="flex w-full h-screen bg-[var(--bg-main)] text-[var(--text-main)] overflow-hidden font-sans select-none relative transition-colors duration-300">
      
      {isImporting && (
         <div className="fixed inset-0 bg-black/80 z-[100] flex flex-col items-center justify-center">
            <div className="w-12 h-12 border-4 border-[#007acc] border-t-transparent rounded-full animate-spin"></div>
            <div className="mt-4 text-white font-bold tracking-widest text-sm">正在处理大规模数据...</div>
         </div>
      )}

      <div className="absolute top-4 left-4 z-20 flex gap-2">
         <label className="flex items-center gap-1.5 px-3 py-1.5 bg-[#007acc] hover:bg-[#0098ff] text-white text-[11px] font-bold rounded shadow-lg cursor-pointer transition-colors border border-[#0055aa]">
            <span>📂</span> 导入 URDF 文件夹
            <input type="file" webkitdirectory="" directory="" multiple className="hidden" onChange={handleFolderUpload} />
         </label>
         
         {robotData && (
           <>
             <label className="flex items-center gap-1.5 px-3 py-1.5 bg-[#2ea043] hover:bg-[#3fb950] text-white text-[11px] font-bold rounded shadow-lg cursor-pointer transition-colors border border-[#238636]">
                <span>📥</span> 导入 CSV
                <input type="file" accept=".csv" className="hidden" onChange={handleImportCSV} />
             </label>
             <button onClick={handleExportCSV} className="flex items-center gap-1.5 px-3 py-1.5 bg-[#2ea043] hover:bg-[#3fb950] text-white text-[11px] font-bold rounded shadow-lg cursor-pointer transition-colors border border-[#238636]">
                <span>📤</span> 导出 CSV
             </button>
             <div className="flex items-center text-[10px] text-[var(--text-main)] bg-[var(--bg-panel)] px-2 rounded border border-[var(--border)] shadow-sm">
                载入: {robotData.name}
             </div>
           </>
         )}
      </div>

      {isURDFModalOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center">
          <div className="bg-[var(--bg-panel)] border border-[var(--border)] p-5 rounded shadow-2xl w-80 flex flex-col gap-4 max-h-[80vh]">
             <h3 className="text-sm font-bold text-[var(--text-highlight)]">检测到多个 URDF 文件</h3>
             <p className="text-xs text-[var(--text-muted)]">请选择要加载的模型：</p>
             <div className="flex-1 overflow-y-auto border border-[var(--border)] rounded p-1 bg-[var(--bg-main)] flex flex-col gap-1 max-h-[40vh]">
                 {availableURDFs.map((file, idx) => (
                     <button key={idx} onClick={() => loadSelectedURDF(file, pendingFiles)} className="w-full text-left px-3 py-2 text-xs text-[var(--text-main)] hover:bg-[#007acc] hover:text-white rounded transition-colors break-all">
                         {file.name}
                     </button>
                 ))}
             </div>
             <div className="flex justify-end gap-2 mt-2">
               <button onClick={() => { setIsURDFModalOpen(false); setPendingFiles([]); }} className="px-3 py-1 bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] border border-[var(--border)] rounded text-xs transition-colors">取消</button>
             </div>
          </div>
        </div>
      )}

      {isFramesModalOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center">
          <div className="bg-[var(--bg-panel)] border border-[var(--border)] p-5 rounded shadow-2xl w-64 flex flex-col gap-4">
             <h3 className="text-sm font-bold text-[var(--text-highlight)]">设置总帧数</h3>
             <input type="number" autoFocus min={10} value={tempFramesInput} onChange={(e) => setTempFramesInput(e.target.value)} className="bg-[var(--bg-main)] border border-[var(--border)] text-[var(--text-main)] p-1.5 rounded outline-none focus:border-[#007acc] w-full" />
             <div className="flex justify-end gap-2 mt-2">
               <button onClick={() => setIsFramesModalOpen(false)} className="px-3 py-1 bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] border border-[var(--border)] rounded text-xs transition-colors">取消</button>
               <button onClick={() => { const val = Math.max(10, parseInt(tempFramesInput) || 60); setTotalFrames(val); setIsFramesModalOpen(false); }} className="px-3 py-1 bg-[#007acc] hover:bg-[#0098ff] text-white rounded text-xs transition-colors">确定</button>
             </div>
          </div>
        </div>
      )}

      <div className="w-[40%] min-w-[350px] flex flex-col border-r border-[var(--border)] bg-[var(--bg-panel)] pt-12">
        <Timeline 
           theme={theme} robotData={robotData} boneNames={activeJoints}
           totalFrames={totalFrames} setTotalFrames={setTotalFrames} onRequestChangeTotalFrames={() => { setTempFramesInput(totalFrames); setIsFramesModalOpen(true); }}
           currentFrame={currentFrame} setCurrentFrame={setCurrentFrame} keyframes={keyframes}
           selectedBones={selectedBones} onSelectBone={setSelectedBones} activeKey={activeKey} setActiveKey={setActiveKey}
           selectionBox={timelineSelection} setSelectionBox={setTimelineSelection} isDraggingTimeline={isDraggingTimeline} setIsDraggingTimeline={setIsDraggingTimeline}
           clipboard={clipboard} onCopy={handleCopyFrames} onPaste={handlePasteFrames} onDelete={handleDeleteSelectedFrames} onSelectBoneRange={handleSelectBoneRange}
           isPlaying={isPlaying} setIsPlaying={setIsPlaying} onStart={() => setCurrentFrame(0)} onEnd={() => setCurrentFrame(totalFrames)}
           onPrevKey={() => { const frames = getAllKeyframeNums(); const prev = frames.reverse().find(f => f < currentFrame); setCurrentFrame(prev !== undefined ? prev : 0); }}
           onNextKey={() => { const frames = getAllKeyframeNums(); const next = frames.find(f => f > currentFrame); setCurrentFrame(next !== undefined ? next : totalFrames); }}
        />
        <CurveEditor curve={currentActiveCurve} theme={theme} onChange={(c) => { if(!activeKey) return; setKeyframes(prev => { const nd = {...prev}; selectedBones.forEach(b => { if(nd[b] && nd[b][activeKey.frame]) nd[b][activeKey.frame].c = c; }); return nd; }); }} disabled={!activeKey || activeKey.frame === 0} />
      </div>

      <div className="flex-1 flex flex-col relative bg-[var(--bg-main)]">
        <div className="absolute top-4 right-4 bg-[var(--bg-panel)] p-1.5 border border-[var(--border)] text-[var(--text-main)] text-[11px] font-sans w-[250px] shadow-lg z-20 flex flex-col gap-1.5 rounded">
          <div className="flex justify-between gap-1 h-6">
             <button onClick={() => setMode('select')} className={`flex-1 border border-[var(--border)] bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] transition-colors ${mode==='select'?'!bg-[var(--accent)] text-white':''}`}>选中</button>
             <button onClick={() => setMode('rotate')} className={`flex-1 border border-[var(--border)] bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] transition-colors ${mode==='rotate'?'!bg-[var(--accent)] text-white':''}`}>旋转</button>
             <button onClick={() => setMode('move')} className={`flex-1 border border-[var(--border)] bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] transition-colors ${mode==='move'?'!bg-[var(--accent)] text-white':''}`}>移动</button>
          </div>
          <div className="flex justify-between gap-1 h-6">
             <button onClick={() => { if(mode==='select') setSelType('box') }} className={`flex-1 border border-[var(--border)] transition-colors ${mode!=='select' ? 'bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed' : (selType==='box' ? '!bg-[var(--accent)] text-white' : 'bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)]')}`}>框选</button>
             <button onClick={selectAll} className={`flex-1 border border-[var(--border)] transition-colors ${mode!=='select' ? 'bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed' : 'bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] active:bg-[var(--accent)]'}`}>选择全部</button>
             <button onClick={() => { if(mode==='select') setSelType('free') }} className={`flex-1 border border-[var(--border)] transition-colors ${mode!=='select' ? 'bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed' : (selType==='free' ? '!bg-[var(--accent)] text-white' : 'bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)]')}`}>自由选择</button>
          </div>
          <div className="flex justify-between gap-1 h-6">
             <button onClick={handleRegisterKeyframe} className="flex-1 border border-[var(--border)] bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)] text-[#e55353] font-bold tracking-widest">注册</button>
             <button onClick={handleReset} className="flex-1 border border-[var(--border)] bg-[var(--bg-btn)] hover:bg-[var(--bg-btn-hover)]">重置</button>
             <button className="flex-1 border border-[var(--border)] bg-[var(--bg-hover)] text-[var(--text-muted)] cursor-not-allowed">物理</button>
          </div>
          <div className="text-center mt-0.5 font-bold text-[10px] text-[var(--text-muted)] border-t border-[var(--border)] pt-1 tracking-widest flex justify-between items-center px-1">
            <span>操作设置</span>
            <div className="flex gap-1">
               <span className="font-normal text-[10px] cursor-pointer hover:scale-110 transition-transform" onClick={()=>setTheme(t => t==='dark'?'light':'dark')} title="切换日夜模式">
                   {theme==='dark'?'🌞':'🌙'}
               </span>
               <span className="font-normal text-[9px] bg-[var(--bg-hover)] px-1 rounded border border-[var(--border)] cursor-pointer hover:bg-[var(--border)] shadow-sm" onClick={()=>setIkMode(m => m==='global'?'selected':'global')} title="全局: IK追溯至根节点&#10;选中: IK仅在已框选的关节之间起效">
                   IK范围: {ikMode==='global'?'全局(链式)':'仅限选中'}
               </span>
               <span className="font-normal text-[9px] bg-[var(--bg-hover)] px-1 rounded border border-[var(--border)] cursor-pointer hover:bg-[var(--border)] shadow-sm" onClick={()=>setSpace(s => s==='local'?'world':'local')}>
                   {space.toUpperCase()}
               </span>
            </div>
          </div>
        </div>

        {robotData && <Viewport3D ref={viewportRef} theme={theme} mode={mode} selType={selType} space={space} ikMode={ikMode} currentFrame={currentFrame} keyframes={keyframes} isPlaying={isPlaying} selectedBones={selectedBones} onSelectBone={setSelectedBones} robotData={robotData} fileMap={fileMap} />}
      </div>
    </div>
  );
}