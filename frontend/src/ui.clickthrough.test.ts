/**
 * Browser click-through: drives the real UI with a real backend and asserts all
 * ten steps work, the mock banner appears, and step 6 is locked until the
 * landmarks are confirmed.
 *
 * Needs both servers up:
 *   backend  : uvicorn main:app --port 8000
 *   frontend : vite            (VITE_UI_URL overrides the address)
 *
 * Skips itself if either is not listening, so it is safe to run anywhere.
 *
 * One test, not ten: the wizard is stateful and each step depends on the last,
 * so splitting it would mean rebuilding the session per test for no gain.
 */
import { chromium, type Locator, type Page } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const FRONTEND = process.env.VITE_UI_URL ?? "http://127.0.0.1:5173";
const BACKEND = "http://127.0.0.1:8000";

/** Flat 200x320 PNG, base64, so the test never writes an image to disk. */
const PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAMgAAAFACAIAAAB2rqTIAAADQklEQVR4nO3SQQkAIADAQPsnsYEZbGUJhyAHF2CPjb0mXDeeF/AlY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCWORMbyJY5EwFgljkTAWCJ0UMHHiA5Ctxidq8QSE07aS1TTGDQ/PRuvsh(np51r9)";

let browser: Awaited<ReturnType<typeof chromium.launch>> | null = null;
let page: Page;
let live = false;

async function exists(loc: Locator, timeout = 15_000): Promise<boolean> {
  try {
    await loc.first().waitFor({ state: "attached", timeout });
    return true;
  } catch {
    return false;
  }
}

async function text(loc: Locator): Promise<string> {
  return (await loc.first().innerText().catch(() => "")) ?? "";
}

/** The stepper button for step n. */
const stepBtn = (n: number) => page.locator(".step-item").nth(n - 1);

async function gotoStep(n: number): Promise<boolean> {
  try {
    await stepBtn(n).click({ timeout: 10_000 });
    return true;
  } catch {
    return false;
  }
}

/** Wait until a step's locked state changes, instead of sleeping a fixed time. */
async function waitUnlocked(n: number, timeout = 10_000): Promise<boolean> {
  try {
    await stepBtn(n).waitFor({ state: "attached" });
    const start = Date.now();
    while (Date.now() - start < timeout) {
      if (!(await stepBtn(n).isDisabled())) return true;
      await page.waitForTimeout(100);
    }
    return false;
  } catch {
    return false;
  }
}

/** Garment buttons carry their design note, so match without anchoring. */
const garmentBtn = (label: string) =>
  page.getByRole("button", { name: new RegExp(label, "i") }).first();

async function attachBoth(): Promise<void> {
  for (const which of ["Front", "Side"]) {
    await page
      .locator(".capture-grid > div", { hasText: `${which} view` })
      .locator('input[type="file"]')
      .setInputFiles({
        name: `${which.toLowerCase()}.png`,
        mimeType: "image/png",
        buffer: Buffer.from(PNG, "base64"),
      });
  }
  await page.waitForTimeout(400);
}

beforeAll(async () => {
  try {
    await fetch(FRONTEND);
    await fetch(`${BACKEND}/api/models`);
    live = true;
  } catch {
    live = false;
    return;
  }
  browser = await chromium.launch();
  page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  await page.goto(FRONTEND, { waitUntil: "networkidle" });
}, 60_000);

afterAll(async () => {
  await browser?.close();
});

describe("the 10 step wizard in a real browser", () => {
  it("walks every step, holds the mock banner, and gates step 6 on landmarks", async () => {
    if (!live) return;

    // ---- step 1
    expect(await exists(garmentBtn("polo shirt"))).toBe(true);
    expect(await exists(page.locator(".canvas svg"))).toBe(true);
    await garmentBtn("polo shirt").click();
    await page.waitForSelector(".mock-banner", { timeout: 10_000 });
    expect(await text(page.locator(".mock-banner"))).toMatch(/mock data/i);
    expect(await text(page.locator(".checklist"))).toMatch(/chest/i);

    // ---- step 2
    expect(await gotoStep(2)).toBe(true);
    await attachBoth();
    await page.getByRole("button", { name: /run capture check/i }).click();
    await page.waitForTimeout(800);
    expect(await page.locator(".check.pass, .check.fail").count()).toBe(4);

    // ---- step 3
    expect(await gotoStep(3)).toBe(true);
    await page.getByRole("button", { name: /^Segment$/ }).click();
    expect(await exists(page.locator(".mask-over"))).toBe(true);
    expect(await text(page.locator(".stat-row"))).toMatch(/mock-silhouette/);

    // ---- step 4
    expect(await gotoStep(4)).toBe(true);
    await page.getByRole("button", { name: /^Calibrate$/ }).click();
    await page.waitForTimeout(500);
    expect(await page.locator("table tbody tr").count()).toBe(2);

    // ---- step 5, and the gate
    expect(await gotoStep(5)).toBe(true);
    await page.getByRole("button", { name: /detect landmarks|reset points/i }).click();
    expect(await exists(page.locator(".lm-wrap canvas"))).toBe(true);
    expect(await stepBtn(6).isDisabled()).toBe(true);

    // pick a point, then click to reposition it: source must become "user"
    await page.locator("#lm-pick").selectOption("waist");
    const box = (await page.locator(".lm-wrap canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.45);
    await page.waitForTimeout(900);
    expect(await page.locator(".lm-key .dot.user").count()).toBe(1);

    await page.getByRole("button", { name: /confirm landmarks/i }).click();
    expect(await waitUnlocked(6)).toBe(true);

    // ---- step 6, both backends
    expect(await gotoStep(6)).toBe(true);
    await page.getByRole("button", { name: /^Regression$/ }).click();
    await page.waitForTimeout(600);
    const rows = await page.locator("table tbody tr").count();
    expect(rows).toBeGreaterThan(0);
    expect(await page.locator(".interval-mid").count()).toBe(rows);

    await page.getByRole("button", { name: /^Body model$/ }).click();
    await page.waitForTimeout(600);
    expect(await page.locator("table tbody tr").count()).toBe(rows);
    expect(await text(page.locator(".stat-row"))).toMatch(/body_model/);

    // ---- step 7
    expect(await gotoStep(7)).toBe(true);
    await page.getByRole("button", { name: /^Validate$/ }).click();
    await page.waitForTimeout(600);
    expect(await page.locator(".flags li").count()).toBeGreaterThan(0);
    expect(await text(page.locator("pre.yaml"))).toMatch(/measurements/);

    const href = await page.getByRole("link", { name: /download yaml/i }).getAttribute("href");
    expect(href).toContain("/export.yaml");
    const res = await fetch(new URL(href!, FRONTEND).toString());
    expect(res.status).toBe(200);
    expect(await res.text()).toMatch(/garment_type: polo_shirt/);

    // ---- step 8, both routes
    expect(await gotoStep(8)).toBe(true);
    await page.getByRole("button", { name: /^Draft$/ }).click();
    await page.waitForTimeout(600);
    const pieces = await page.locator(".piece").count();
    expect(pieces).toBeGreaterThan(0);
    const blockSvg = await page.locator(".piece svg").first().innerHTML();

    await page.getByRole("button", { name: /^GarmentCode$/ }).click();
    await page.waitForTimeout(700);
    expect(await page.locator(".piece").count()).toBe(pieces);
    // same geometry, different route treatment
    expect(await page.locator(".piece svg").first().innerHTML()).not.toBe(blockSvg);

    // ---- step 9
    expect(await gotoStep(9)).toBe(true);
    await page.getByRole("button", { name: /add production details/i }).click();
    await page.waitForTimeout(700);
    expect(await page.locator(".piece-meta").count()).toBeGreaterThan(0);
    expect(await text(page.locator(".piece-meta"))).toMatch(/SA .*cm/);

    // ---- step 10
    expect(await gotoStep(10)).toBe(true);
    await page.getByRole("button", { name: /prepare toile/i }).click();
    await page.waitForTimeout(600);
    const items = page.locator(".toile-list li");
    expect(await items.count()).toBeGreaterThan(0);
    await items.first().locator("input").check();
    await page.waitForTimeout(200);
    expect(await text(page.locator(".notice").last())).toMatch(/1 of/);

    // the banner never left: everything in this session is mock
    expect(await exists(page.locator(".mock-banner"))).toBe(true);
  }, 180_000);

  it("switching garment changes what step 6 reports", async () => {
    if (!live) return;

    expect(await gotoStep(1)).toBe(true);
    await garmentBtn("^slacks").click();
    await page.waitForTimeout(900);

    expect(await gotoStep(6)).toBe(true);
    await page.getByRole("button", { name: /^Measure$/ }).click();
    await page.waitForTimeout(700);

    const names = (await page.locator("table tbody tr td:first-child").allInnerTexts()).join(" ");
    expect(names).toMatch(/inseam/);
    expect(names).not.toMatch(/sleeve/);
  }, 60_000);

  it("sample photos unlock the pipeline without an upload", async () => {
    if (!live) return;

    // fresh session so the gate starts locked again
    await page.goto(FRONTEND, { waitUntil: "networkidle" });
    await page.waitForTimeout(600);

    expect(await stepBtn(6).isDisabled()).toBe(true);

    // step 1 first: step 6 reads the garment spec to know which measurements
    // it should report
    expect(await gotoStep(1)).toBe(true);
    await garmentBtn("polo shirt").click();
    await page.waitForTimeout(500);

    expect(await gotoStep(2)).toBe(true);
    await page.getByRole("button", { name: /use sample photos/i }).click();
    await page.waitForTimeout(500);
    expect(await page.locator(".drop img").count()).toBe(2);

    await page.getByRole("button", { name: /run capture check/i }).click();
    await page.waitForTimeout(700);
    expect(await page.locator(".check.pass, .check.fail").count()).toBe(4);

    expect(await gotoStep(5)).toBe(true);
    await page.getByRole("button", { name: /detect landmarks|reset points/i }).click();
    expect(await exists(page.locator(".lm-wrap canvas"))).toBe(true);

    await page.getByRole("button", { name: /confirm landmarks/i }).click();
    expect(await waitUnlocked(6)).toBe(true);

    // 7 to 10 need step 6 to have run as well, not just the confirmation
    expect(await stepBtn(10).isDisabled()).toBe(true);
    expect(await gotoStep(6)).toBe(true);
    await page.getByRole("button", { name: /^Regression$/ }).click();
    expect(await waitUnlocked(10)).toBe(true);
  }, 90_000);
});
