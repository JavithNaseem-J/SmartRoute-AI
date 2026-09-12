export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function formatCurrency(value: unknown) {
  const amount = typeof value === "number" ? value : Number(value || 0);
  return `$${amount.toFixed(4)}`;
}
