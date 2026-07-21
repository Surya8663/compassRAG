import { redirect } from "next/navigation";

export default function RootPage() {
  // Redirect right to the centerpiece chat/query interface
  redirect("/chat");
}
