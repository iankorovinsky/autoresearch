import { NextRequest, NextResponse } from "next/server";
import { createClient } from "../../../lib/supabase/server";
import { appOrigin } from "../../../lib/auth/config";
import { authDestination } from "../../../lib/auth/navigation";

export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code");
  if (code && !request.nextUrl.searchParams.has("error")) {
    const supabase = await createClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) {
      return NextResponse.redirect(
        new URL(
          authDestination(request.nextUrl.searchParams.get("next")),
          appOrigin(),
        ),
        { headers: { "Cache-Control": "no-store" } },
      );
    }
  }
  return NextResponse.redirect(
    new URL("/auth/login?error=callback", appOrigin()),
    { headers: { "Cache-Control": "no-store" } },
  );
}
