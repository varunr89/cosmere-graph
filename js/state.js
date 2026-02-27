// js/state.js -- Shared mutable state across modules

import { GEM_COLORS } from './constants.js';

// Data loaded during init
export let graph = null;
export let entries = null;
export let similarity = {};

// D3 force simulation
export let simulation = null;

// Currently focused node
export let focusedNode = null;

// Active type filters
export let activeFilters = new Set(Object.keys(GEM_COLORS));

// Embedding-related
export let scoresData = null;
export let explicitTagsByEntry = {};
export let baselineConnected = {};

// Model selection
export let activeModel = null;
export let modelsManifest = null;
export let scoresCache = {};

// Hypothesis layer
export let implicitLinkGroup = null;
export let implicitLinks = null;
export let _originalTick = null;

// Review decisions: key "entity::entryId" -> "confirmed" | "rejected" | undefined
export let reviewState = {};

// Mobile backdrop DOM refs
export let _panelBackdrop = null;
export let _reviewBackdrop = null;

// Setters (ES module exports are read-only bindings)
export function setGraph(val) { graph = val; }
export function setEntries(val) { entries = val; }
export function setSimilarity(val) { similarity = val; }
export function setSimulation(val) { simulation = val; }
export function setFocusedNode(val) { focusedNode = val; }
export function setScoresData(val) { scoresData = val; }
export function setActiveModel(val) { activeModel = val; }
export function setModelsManifest(val) { modelsManifest = val; }
export function setImplicitLinkGroup(val) { implicitLinkGroup = val; }
export function setImplicitLinks(val) { implicitLinks = val; }
export function set_originalTick(val) { _originalTick = val; }
export function set_panelBackdrop(val) { _panelBackdrop = val; }
export function set_reviewBackdrop(val) { _reviewBackdrop = val; }
