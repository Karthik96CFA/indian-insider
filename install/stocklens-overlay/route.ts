import { NextResponse } from 'next/server'
import { db } from '@/server/db'
import { signalEvents } from '@/server/db/schema'
import { eq, and, gt } from 'drizzle-orm'

export async function POST(req: Request) {
  try {
    const integrationSecret = process.env.INTEGRATION_SECRET?.trim()
    if (integrationSecret) {
      const provided = req.headers.get('X-Integration-Secret')?.trim()
      if (provided !== integrationSecret) {
        return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 })
      }
    }

    const body = await req.json()
    const {
      symbol,
      event_type,
      category,
      severity,
      signal_strength,
      confidence,
      source_engine,
      payload_json,
      event_hash,
      cooldown_minutes,
      ttl_days,
    } = body

    if (!symbol || !event_type || !category || !severity || !confidence || !source_engine || !event_hash) {
      return NextResponse.json({ success: false, error: 'Missing mandatory payload properties' }, { status: 400 })
    }

    // 1. Idempotency Check — drop duplicate webhook requests immediately
    const existing = await db
      .select({ id: signalEvents.id })
      .from(signalEvents)
      .where(eq(signalEvents.eventHash, event_hash))
      .limit(1)

    if (existing.length > 0) {
      return NextResponse.json({ success: true, message: 'Duplicate signal dropped' })
    }

    // 2. Temporal Deduplication Window Check
    // e.g., suppress identical accumulation scans on the same symbol for 24h
    const cooldown = cooldown_minutes || 1440 // Default to 24 hours (1440 minutes)
    const cutoff = new Date(Date.now() - cooldown * 60 * 1000)
    
    const recentSuppression = await db
      .select({ id: signalEvents.id })
      .from(signalEvents)
      .where(
        and(
          eq(signalEvents.symbol, symbol),
          eq(signalEvents.eventType, event_type),
          gt(signalEvents.createdAt, cutoff)
        )
      )
      .limit(1)

    if (recentSuppression.length > 0) {
      return NextResponse.json({ success: true, message: 'Signal suppressed inside temporal deduplication window' })
    }

    // 3. Calculate explicit signal TTL expiry (prevents permanent alert clutter)
    const expiresAt = new Date()
    expiresAt.setDate(expiresAt.getDate() + (ttl_days || 10))

    // 4. Write event directly to database — keep receiver thin & fast
    await db.insert(signalEvents).values({
      symbol,
      eventType: event_type,
      category,
      severity,
      signalStrength: signal_strength,
      confidence,
      sourceEngine: source_engine,
      payloadJson: payload_json,
      eventHash: event_hash,
      cooldownMinutes: cooldown,
      expiresAt,
    })

    return NextResponse.json({ success: true })
  } catch (error: unknown) {
    return NextResponse.json({ success: false, error: (error as Error).message }, { status: 500 })
  }
}
