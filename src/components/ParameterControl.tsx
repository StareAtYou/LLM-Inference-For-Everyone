import { useId } from "react";
export default function ParameterControl({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  unit = "",
  disabled = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
  unit?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="parameter-control">
      <div>
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>
          {value.toLocaleString()} {unit}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="parameter-bounds mono">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
