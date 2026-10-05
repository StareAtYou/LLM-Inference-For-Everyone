import type { PrefixNode } from "../simulation/prefix";
function Branch({ node }: { node: PrefixNode }) {
  return (
    <li>
      <span className={node.count > 1 ? "shared" : ""}>
        {node.token} <small>×{node.count}</small>
      </span>
      {node.children.length > 0 && (
        <ul>
          {node.children.map((c) => (
            <Branch node={c} key={c.token} />
          ))}
        </ul>
      )}
    </li>
  );
}
export default function PrefixTree({ tree }: { tree: PrefixNode }) {
  return (
    <div
      className="prefix-tree"
      role="img"
      aria-label="Token 前缀树，共同开头共享节点"
    >
      <ul>
        <Branch node={tree} />
      </ul>
      <p className="dark-description">
        紫色节点被多个请求共享。此处是一 Token 一节点的教学树，真实 Radix
        节点可压缩多个 Token。
      </p>
    </div>
  );
}
