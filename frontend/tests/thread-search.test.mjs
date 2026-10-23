import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createSearchIndex, searchThreads } from '../src/lib/search/thread-search.ts';

const threads=[
  {id:'matrix',title:'Matrix tiling',task:'Optimize multiplication',project:'GPU research',messages:[{role:'user',text:'Compare T4 latency against the baseline.'},{role:'assistant',text:'Register pressure limits occupancy; try walnut as a test keyword.'}]},
  {id:'softmax',title:'Fused softmax',task:'Fuse memory operations',project:'GPU research',messages:[{role:'user',text:'Review matrix tiling on the new candidate.'}]},
  {id:'accent',title:'Résumé of experiments',task:'Review',project:'Notes',messages:[]},
];
const index=createSearchIndex(threads);

test('empty queries show threads in workspace order, without message duplicates',()=>{
  assert.deepEqual(searchThreads(index,'  ').map(result=>result.id),['thread:matrix','thread:softmax','thread:accent']);
});
test('exact thread titles rank ahead of message references',()=>{
  const results=searchThreads(index,'matrix tiling');
  assert.equal(results[0].id,'thread:matrix');
  assert.ok(results.some(result=>result.id==='message:softmax:0'));
});
test('message-only matches retain the correct thread and message location',()=>{
  const results=searchThreads(index,'walnut');
  assert.equal(results.length,1);
  assert.equal(results[0].threadId,'matrix');
  assert.equal(results[0].messageIndex,1);
  assert.match(results[0].snippet,/walnut/);
});
test('matching ignores case, accents, and repeated whitespace; all query words are required',()=>{
  assert.equal(searchThreads(index,' RESUME   EXPERIMENTS ')[0].threadId,'accent');
  assert.equal(searchThreads(index,'latency T4')[0].id,'message:matrix:0');
  assert.deepEqual(searchThreads(index,'latency zebraplatypus'),[]);
});
test('fuzzy matching supports ordered character abbreviations and safely handles punctuation',()=>{
  assert.equal(searchThreads(index,'fsd sfmx')[0].threadId,'softmax');
  assert.deepEqual(searchThreads(index,'[.*]+'),[]);
});
test('index changes are isolated and snippets show distant matches',()=>{
  const fresh=createSearchIndex([{...threads[0],messages:[{role:'user',text:'prefix '.repeat(100)+'needle at the end'}]}]);
  const result=searchThreads(fresh,'needle')[0];
  assert.ok(result.snippet.startsWith('…'));
  assert.match(result.snippet,/needle/);
  assert.ok(result.snippet.length<=147);
  assert.deepEqual(searchThreads(index,'needle'),[]);
});
