export type PrefixNode = {
  token: string;
  count: number;
  children: PrefixNode[];
};
export function buildPrefixTree(sequences: string[][]): PrefixNode {
  const root: PrefixNode = {
    token: "ROOT",
    count: sequences.length,
    children: [],
  };
  for (const seq of sequences) {
    let node = root;
    for (const token of seq) {
      let child = node.children.find((c) => c.token === token);
      if (!child) {
        child = { token, count: 0, children: [] };
        node.children.push(child);
      }
      child.count++;
      node = child;
    }
  }
  return root;
}
