import assert from 'node:assert/strict';
import {test} from 'node:test';
import {sampleExperiments,metricValue,bestHistory,clampRange,nearestPoint,chartDomain,toggleComparison,experimentsCsv} from '../src/lib/charts/experiments.ts';
test('speedup uses each measured baseline and best-so-far does not reset with viewport',()=>{
 const points=[3,2,2.5,1].map(runtime=>({runtime,baseline:3}));
 assert.equal(metricValue(points[3],'speedup'),3);
 assert.deepEqual(bestHistory(points,'runtime'),[3,2,2,1]);
 assert.deepEqual(bestHistory(points,'speedup').slice(2),[1.5,3]);
});
test('zoom clamps bounds and always retains two experiments, including reversed drags',()=>{
 assert.deepEqual(clampRange(8,2,12),[2,8]);
 assert.deepEqual(clampRange(11,11,12),[10,11]);
 assert.deepEqual(clampRange(-3,30,12),[0,11]);
 assert.equal(nearestPoint(-1,3,8),3);
 assert.equal(nearestPoint(2,3,8),8);
 assert.equal(nearestPoint(.4,3,8),5);
});
test('comparison toggles and retains only the two most recent candidates',()=>{
 assert.deepEqual(toggleComparison([0,4],6),[4,6]);
 assert.deepEqual(toggleComparison([0,4],0),[4]);
});
test('chart domains include a stable range for constant values',()=>{
 const [low,high]=chartDomain([1,1]);
 assert.ok(low<1&&high>1);
 assert.ok(chartDomain([0])[1]>0);
});
test('export contains only supplied viewport measurements, quotes text and marks sample data',()=>{
 const csv=experimentsCsv(sampleExperiments.slice(9));
 assert.equal(csv.trim().split('\r\n').length,4);
 assert.ok(csv.includes('EXP-010')&&csv.includes('EXP-012'));
 assert.ok(!csv.includes('EXP-009'));
 assert.match(csv,/"sample"/);
 assert.match(experimentsCsv([{...sampleExperiments[0],configuration:'tile, "large"'}]),/"tile, ""large"""/);
});
