export type Depth = "beginner" | "advanced" | "expert";
export type ModelId = "qwen38-dense" | "qwen36-moe" | "teaching";
export type SourceRef = {
  id: string;
  title: string;
  url: string;
  checkedAt: string;
  revision?: string;
  path?: string;
  symbol?: string;
  startLine?: number;
};
export type ModelPreset = {
  id: ModelId;
  name: string;
  family: "dense" | "moe";
  hiddenSize: number;
  headDim: number;
  qHeads: number;
  kvHeads: number;
  layerTypes: ("linear_attention" | "full_attention")[];
  ffnSize?: number;
  expertCount?: number;
  activeExperts?: number;
  sharedExperts?: number;
  linearKeyHeads: number;
  linearValueHeads: number;
  linearHeadDim: number;
  sources: SourceRef[];
};
export type Topic = {
  id: string;
  title: string;
  englishTerms: string[];
  area: string;
  prerequisites: string[];
  levels: Record<
    Depth,
    { summary: string; explanation: string; details: string[] }
  >;
  links: { label: string; to: string }[];
  sources: SourceRef[];
};
export type Preferences = {
  version: 1;
  depth: Depth;
  bookmarks: string[];
  recent: string[];
};
export type Observation = {
  label: string;
  value: number;
  unit: string;
  source: "simulation" | "estimate" | "measured";
  assumptions: string[];
};
