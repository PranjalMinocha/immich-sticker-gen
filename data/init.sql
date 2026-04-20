CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS "sticker_generation" (
    "id" uuid NOT NULL DEFAULT uuid_generate_v4(),
    "source" varchar(16) NOT NULL DEFAULT 'synthetic',
    "userId" uuid NOT NULL,
    "assetId" uuid NOT NULL,
    "bbox" jsonb NOT NULL,
    "pointCoords" jsonb NOT NULL,
    "mlSuggestedMask" text NOT NULL,
    "userSavedMask" text,
    "s3StickerKey" varchar(255),
    "processingTimeMs" integer NOT NULL DEFAULT 0,
    "saved" boolean,
    "numTries" integer NOT NULL DEFAULT 1,
    "editedPixels" integer NOT NULL DEFAULT 0,
    "qualityStatus" varchar(16) NOT NULL DEFAULT 'pending',
    "qualityCheckedAt" timestamptz,
    "qualityCheckVersion" integer NOT NULL DEFAULT 1,
    "qualityFailReasonsJson" text,
    "usedForTraining" boolean NOT NULL DEFAULT false,
    "usedForTrainingAt" timestamptz,
    "retrainRunId" varchar(64),
    "createdAt" timestamptz NOT NULL DEFAULT now(),
    "updatedAt" timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT "sticker_generation_source_check" CHECK ("source" IN ('synthetic', 'real')),
    CONSTRAINT "sticker_generation_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "sticker_generation_userId_fkey" FOREIGN KEY ("userId") REFERENCES "user" ("id") ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT "sticker_generation_assetId_fkey" FOREIGN KEY ("assetId") REFERENCES "asset" ("id") ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS "sticker_generation_source_idx" ON "sticker_generation" ("source");
CREATE INDEX IF NOT EXISTS "sticker_generation_userId_idx" ON "sticker_generation" ("userId");
CREATE INDEX IF NOT EXISTS "sticker_generation_assetId_idx" ON "sticker_generation" ("assetId");
CREATE INDEX IF NOT EXISTS "sticker_generation_saved_idx" ON "sticker_generation" ("saved");
CREATE INDEX IF NOT EXISTS "sticker_generation_createdAt_idx" ON "sticker_generation" ("createdAt");
