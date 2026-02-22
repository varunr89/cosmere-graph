/**
 * Tagging Engine -- pure-function recomputation engine for implicit tags.
 *
 * Takes scores.json data and user-controlled settings, produces a set of
 * implicit tags and rebuilt co-occurrence edges. No DOM dependencies -- all
 * functions are testable standalone.
 *
 * scores.json structure per entity:
 * {
 *   "entities": {
 *     "kaladin": {
 *       "specificity": 4.2,
 *       "prototypes": 2,
 *       "calibration": { "p10": 0.72, "p15": 0.68, ..., "p50": 0.45 },
 *       "scores": { "1396": [0.78, 0.55], "1285": [0.65, 0.72], ... }
 *     },
 *     ...
 *   }
 * }
 *
 * Scores may be arrays (multi-prototype, take max) or plain numbers (single value).
 */

// ---- Pure functions --------------------------------------------------------

/**
 * Resolve a score value to a single number. In multi-prototype mode, scores
 * are arrays of per-prototype similarity values; the effective score is the
 * maximum. In single-prototype mode, the score is already a number.
 *
 * @param {number|Array<number>} scoreValue - A score or array of scores
 * @returns {number} The effective score
 */
function resolveScore(scoreValue) {
  if (typeof scoreValue === 'number') return scoreValue;
  if (Array.isArray(scoreValue)) {
    var best = -Infinity;
    for (var i = 0; i < scoreValue.length; i++) {
      if (scoreValue[i] > best) best = scoreValue[i];
    }
    return best;
  }
  return Number(scoreValue) || 0;
}

/**
 * Compute the effective threshold for an entity based on calibration stats.
 * The calibration object stores percentile thresholds computed offline from
 * the score distribution of explicitly-tagged entries. p10 is the 10th
 * percentile (low score, loose threshold), p50 is the median (higher score,
 * stricter threshold). Higher percentile = stricter threshold = fewer tags.
 *
 * @param {Object} entityData - Entity object from scores.json (has .calibration)
 * @param {number} calibrationPercentile - One of 10, 15, 20, 25, 30, 35, 40, 45, 50
 * @returns {number} The threshold for this entity
 */
function computeEffectiveThreshold(entityData, calibrationPercentile) {
  if (!entityData || !entityData.calibration) {
    throw new Error('entityData must have a calibration object');
  }
  var key = 'p' + calibrationPercentile;
  var threshold = entityData.calibration[key];
  if (threshold === undefined) {
    throw new Error('No calibration value for percentile ' + calibrationPercentile);
  }
  return threshold;
}

/**
 * Filter entities by specificity score. Specificity measures how topically
 * focused an entity is -- hub entities (e.g., "allomancy" with 397 entries)
 * have low specificity and produce noisy implicit tags. This filter removes
 * them.
 *
 * @param {Object} entities - The scores.json "entities" object
 * @param {number} minSpecificity - Minimum specificity to include
 * @returns {Object} Filtered entities object (shallow copy, same value refs)
 */
function filterBySpecificity(entities, minSpecificity) {
  var filtered = {};
  var keys = Object.keys(entities);
  for (var i = 0; i < keys.length; i++) {
    var name = keys[i];
    var entity = entities[name];
    if (entity.specificity >= minSpecificity) {
      filtered[name] = entity;
    }
  }
  return filtered;
}

/**
 * Apply margin constraint: for each entry, reject tags where the score falls
 * too far below the best-scoring entity for that entry. This prevents an
 * entry from being tagged with a marginally-above-threshold entity when
 * another entity is a much stronger match.
 *
 * @param {Array} candidateTags - [{entity, entryId, score}, ...]
 * @param {number} margin - Confidence margin (0.0 to 0.15). Tags where
 *   (bestScore - thisScore) > margin are removed.
 * @returns {Array} Filtered tags (new array, original untouched)
 */
function applyMarginFilter(candidateTags, margin) {
  if (margin <= 0) {
    return candidateTags.slice();
  }

  // Group by entryId to find best score per entry
  var byEntry = {};
  for (var i = 0; i < candidateTags.length; i++) {
    var tag = candidateTags[i];
    if (!byEntry[tag.entryId]) {
      byEntry[tag.entryId] = [];
    }
    byEntry[tag.entryId].push(tag);
  }

  var result = [];
  var entryIds = Object.keys(byEntry);
  for (var j = 0; j < entryIds.length; j++) {
    var entryId = entryIds[j];
    var tags = byEntry[entryId];
    // Find best score for this entry
    var bestScore = -Infinity;
    for (var k = 0; k < tags.length; k++) {
      if (tags[k].score > bestScore) {
        bestScore = tags[k].score;
      }
    }
    // Keep only tags within margin of best
    for (var k = 0; k < tags.length; k++) {
      if ((bestScore - tags[k].score) <= margin) {
        result.push(tags[k]);
      }
    }
  }

  return result;
}

/**
 * Apply must-bridge filter: only keep implicit tags on entries that already
 * have at least one other tag (explicit or implicit from a different entity).
 * This prevents orphan implicit tags that would create disconnected edges.
 *
 * @param {Array} implicitTags - [{entity, entryId, score}, ...]
 * @param {Object} explicitTagsByEntry - {entryId: Set of entity names}
 * @returns {Array} Filtered tags (new array)
 */
function applyMustBridgeFilter(implicitTags, explicitTagsByEntry) {
  // Build a set of entries that have at least one explicit tag
  var entriesWithExplicitTags = {};
  var entryIds = Object.keys(explicitTagsByEntry);
  for (var i = 0; i < entryIds.length; i++) {
    var entryId = entryIds[i];
    var tags = explicitTagsByEntry[entryId];
    // Accept both Set and Array
    var size = tags.size !== undefined ? tags.size : tags.length;
    if (size > 0) {
      entriesWithExplicitTags[entryId] = true;
    }
  }

  // Also count how many distinct entities appear in implicit tags per entry
  var implicitEntitiesByEntry = {};
  for (var i = 0; i < implicitTags.length; i++) {
    var tag = implicitTags[i];
    if (!implicitEntitiesByEntry[tag.entryId]) {
      implicitEntitiesByEntry[tag.entryId] = {};
    }
    implicitEntitiesByEntry[tag.entryId][tag.entity] = true;
  }

  var result = [];
  for (var i = 0; i < implicitTags.length; i++) {
    var tag = implicitTags[i];
    var hasExplicit = !!entriesWithExplicitTags[tag.entryId];
    // Count other implicit entities on this entry (excluding this one)
    var otherImplicitCount = 0;
    if (implicitEntitiesByEntry[tag.entryId]) {
      var implicitEntities = Object.keys(implicitEntitiesByEntry[tag.entryId]);
      otherImplicitCount = implicitEntities.length - 1; // exclude self
    }
    // Keep if entry has explicit tags OR has other implicit entities
    if (hasExplicit || otherImplicitCount > 0) {
      result.push(tag);
    }
  }

  return result;
}

/**
 * Rebuild co-occurrence edges from combined explicit + implicit tags.
 * Two entities share an edge if they co-occur on the same entry. Edges from
 * implicit tags get type="implicit"; edges from explicit-only tags get
 * type="explicit"; edges with both get type="mixed".
 *
 * @param {Object} explicitTagsByEntry - {entryId: Set of entity names}
 * @param {Array} implicitTags - [{entity, entryId, score}, ...]
 * @param {number} minEdgeWeight - Minimum co-occurrence count to keep edge (default 2)
 * @returns {Object} {
 *   edges: [{source, target, weight, type, entryIds}],
 *   stats: {totalImplicitTags, totalEdges, implicitEdges, mixedEdges, explicitEdges, rescuedNodes}
 * }
 */
function rebuildEdges(explicitTagsByEntry, implicitTags, minEdgeWeight) {
  if (minEdgeWeight === undefined) minEdgeWeight = 2;

  // Merge explicit and implicit tags per entry
  // Track which entity-entry pairs are implicit-only for edge typing
  // Skip implicit tags that duplicate an explicit tag on the same entry
  var implicitPairs = {};
  for (var i = 0; i < implicitTags.length; i++) {
    var tag = implicitTags[i];
    var explicitTags = explicitTagsByEntry[tag.entryId];
    var isAlsoExplicit = false;
    if (explicitTags) {
      if (explicitTags.has) {
        isAlsoExplicit = explicitTags.has(tag.entity);
      } else if (Array.isArray(explicitTags)) {
        isAlsoExplicit = explicitTags.indexOf(tag.entity) !== -1;
      }
    }
    if (!isAlsoExplicit) {
      implicitPairs[tag.entity + '::' + tag.entryId] = true;
    }
  }

  // Build combined tags per entry
  var combinedByEntry = {};
  // Start with explicit tags
  var entryIds = Object.keys(explicitTagsByEntry);
  for (var i = 0; i < entryIds.length; i++) {
    var entryId = entryIds[i];
    var tags = explicitTagsByEntry[entryId];
    // Convert Set to array if needed
    var tagArray = tags.forEach ? [] : tags;
    if (tags.forEach && tags.size !== undefined) {
      tags.forEach(function(t) { tagArray.push(t); });
    } else if (Array.isArray(tags)) {
      tagArray = tags.slice();
    }
    combinedByEntry[entryId] = tagArray.slice();
  }
  // Add implicit tags
  for (var i = 0; i < implicitTags.length; i++) {
    var tag = implicitTags[i];
    if (!combinedByEntry[tag.entryId]) {
      combinedByEntry[tag.entryId] = [];
    }
    // Avoid duplicates (entity might already be in explicit tags)
    if (combinedByEntry[tag.entryId].indexOf(tag.entity) === -1) {
      combinedByEntry[tag.entryId].push(tag.entity);
    }
  }

  // Build edge map: "entityA||entityB" -> {weight, entryIds, hasExplicit, hasImplicit}
  var edgeMap = {};
  var allEntryIds = Object.keys(combinedByEntry);
  for (var i = 0; i < allEntryIds.length; i++) {
    var entryId = allEntryIds[i];
    var tags = combinedByEntry[entryId];
    // Generate all pairs
    for (var a = 0; a < tags.length; a++) {
      for (var b = a + 1; b < tags.length; b++) {
        var entityA = tags[a];
        var entityB = tags[b];
        // Canonical order for consistent keys
        var source = entityA < entityB ? entityA : entityB;
        var target = entityA < entityB ? entityB : entityA;
        var edgeKey = source + '||' + target;

        if (!edgeMap[edgeKey]) {
          edgeMap[edgeKey] = {
            source: source,
            target: target,
            weight: 0,
            entryIds: [],
            hasExplicit: false,
            hasImplicit: false
          };
        }
        edgeMap[edgeKey].weight++;
        edgeMap[edgeKey].entryIds.push(entryId);

        // Determine if this co-occurrence involves an implicit tag
        var aIsImplicit = !!implicitPairs[entityA + '::' + entryId];
        var bIsImplicit = !!implicitPairs[entityB + '::' + entryId];
        if (aIsImplicit || bIsImplicit) {
          edgeMap[edgeKey].hasImplicit = true;
        }
        if (!aIsImplicit || !bIsImplicit) {
          // At least one side is explicit on this entry
          edgeMap[edgeKey].hasExplicit = true;
        }
      }
    }
  }

  // Filter by minimum weight and build result
  var edges = [];
  var edgeKeys = Object.keys(edgeMap);
  var implicitEdgeCount = 0;
  var mixedEdgeCount = 0;
  var explicitEdgeCount = 0;

  // Track which entities appear in edges (for rescued node count)
  var connectedEntities = {};

  for (var i = 0; i < edgeKeys.length; i++) {
    var e = edgeMap[edgeKeys[i]];
    if (e.weight < minEdgeWeight) continue;

    var type;
    if (e.hasImplicit && e.hasExplicit) {
      type = 'mixed';
      mixedEdgeCount++;
    } else if (e.hasImplicit) {
      type = 'implicit';
      implicitEdgeCount++;
    } else {
      type = 'explicit';
      explicitEdgeCount++;
    }

    edges.push({
      source: e.source,
      target: e.target,
      weight: e.weight,
      type: type,
      entryIds: e.entryIds
    });

    connectedEntities[e.source] = true;
    connectedEntities[e.target] = true;
  }

  return {
    edges: edges,
    stats: {
      totalImplicitTags: implicitTags.length,
      totalEdges: edges.length,
      implicitEdges: implicitEdgeCount,
      mixedEdges: mixedEdgeCount,
      explicitEdges: explicitEdgeCount,
      connectedEntities: Object.keys(connectedEntities).length
    }
  };
}

/**
 * Main entry point: compute all implicit tags given scores and settings.
 *
 * Pipeline:
 * 1. Filter entities by specificity
 * 2. For each surviving entity, compute threshold from calibration percentile
 * 3. For each entry in that entity's scores, if score >= threshold, create candidate tag
 * 4. Exclude entries that already have this entity as an explicit tag
 * 5. Apply margin filter
 * 6. Optionally apply must-bridge filter
 * 7. Rebuild edges from explicit + implicit tags
 *
 * @param {Object} scoresData - Full scores.json object (has .entities)
 * @param {Object} explicitTagsByEntry - {entryId: Set of entity names}
 * @param {Object} baselineConnected - Set of entity IDs connected in explicit-only graph
 *   (used for rescued-node stats)
 * @param {Object} settings - {
 *   calibrationPercentile: number (10-50, default 25),
 *   minSpecificity: number (0-10, default 0),
 *   confidenceMargin: number (0.0-0.15, default 0),
 *   mustBridge: boolean (default false)
 * }
 * @returns {Object} {
 *   implicitTags: [{entity, entryId, score}, ...],
 *   edges: [{source, target, weight, type, entryIds}, ...],
 *   stats: {totalTags, rescuedNodes, ...}
 * }
 */
function computeImplicitTags(scoresData, explicitTagsByEntry, baselineConnected, settings) {
  // Default settings
  var calibrationPercentile = (settings && settings.calibrationPercentile !== undefined)
    ? settings.calibrationPercentile : 25;
  var minSpecificity = (settings && settings.minSpecificity !== undefined) ? settings.minSpecificity : 0;
  var confidenceMargin = (settings && settings.confidenceMargin !== undefined) ? settings.confidenceMargin : 0;
  var mustBridge = (settings && settings.mustBridge) || false;
  var minEdgeWeight = (settings && settings.minEdgeWeight !== undefined)
    ? settings.minEdgeWeight : 2;

  // Step 1: Filter by specificity
  var entities = filterBySpecificity(scoresData.entities, minSpecificity);

  // Step 2-4: Generate candidate implicit tags
  var candidates = [];
  var entityNames = Object.keys(entities);
  for (var i = 0; i < entityNames.length; i++) {
    var entityName = entityNames[i];
    var entityData = entities[entityName];

    // Compute threshold for this entity
    var threshold = computeEffectiveThreshold(entityData, calibrationPercentile);

    // Scan all scored entries
    if (!entityData.scores) continue;
    var entryIds = Object.keys(entityData.scores);
    for (var j = 0; j < entryIds.length; j++) {
      var entryId = entryIds[j];
      var score = resolveScore(entityData.scores[entryId]);

      // Skip if below threshold
      if (score < threshold) continue;

      // Skip if this entry already has this entity as an explicit tag
      var explicitTags = explicitTagsByEntry[entryId];
      if (explicitTags) {
        var hasExplicit = false;
        if (explicitTags.has) {
          hasExplicit = explicitTags.has(entityName);
        } else if (Array.isArray(explicitTags)) {
          hasExplicit = explicitTags.indexOf(entityName) !== -1;
        }
        if (hasExplicit) continue;
      }

      candidates.push({
        entity: entityName,
        entryId: entryId,
        score: score
      });
    }
  }

  // Step 5: Apply margin filter
  if (confidenceMargin > 0) {
    candidates = applyMarginFilter(candidates, confidenceMargin);
  }

  // Step 6: Apply must-bridge filter
  if (mustBridge) {
    candidates = applyMustBridgeFilter(candidates, explicitTagsByEntry);
  }

  // Step 7: Rebuild edges
  var edgeResult = rebuildEdges(explicitTagsByEntry, candidates, minEdgeWeight);

  // Compute rescued nodes: entities connected in new graph but not in baseline
  var rescuedNodes = 0;
  if (baselineConnected) {
    var newConnected = edgeResult.stats.connectedEntities;
    // baselineConnected might be a Set or a plain object
    var baselineCheck = baselineConnected.has
      ? function(id) { return baselineConnected.has(id); }
      : function(id) { return !!baselineConnected[id]; };

    // We need to iterate the connected entities from edges
    var connectedSet = {};
    for (var i = 0; i < edgeResult.edges.length; i++) {
      connectedSet[edgeResult.edges[i].source] = true;
      connectedSet[edgeResult.edges[i].target] = true;
    }
    var connectedIds = Object.keys(connectedSet);
    for (var i = 0; i < connectedIds.length; i++) {
      if (!baselineCheck(connectedIds[i])) {
        rescuedNodes++;
      }
    }
  }

  return {
    implicitTags: candidates,
    edges: edgeResult.edges,
    stats: {
      totalTags: candidates.length,
      totalEdges: edgeResult.edges.length,
      implicitEdges: edgeResult.stats.implicitEdges,
      mixedEdges: edgeResult.stats.mixedEdges,
      explicitEdges: edgeResult.stats.explicitEdges,
      rescuedNodes: rescuedNodes,
      connectedEntities: edgeResult.stats.connectedEntities,
      entitiesConsidered: entityNames.length
    }
  };
}

// Export for Node.js (Playwright tests) while staying compatible with browsers
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    resolveScore: resolveScore,
    computeEffectiveThreshold: computeEffectiveThreshold,
    filterBySpecificity: filterBySpecificity,
    applyMarginFilter: applyMarginFilter,
    applyMustBridgeFilter: applyMustBridgeFilter,
    rebuildEdges: rebuildEdges,
    computeImplicitTags: computeImplicitTags
  };
}
