import type {
  ExecuteJobResult,
  ValidationResult,
} from "../../../runtime/offeringTypes.js";

// Verify server endpoint — set via env or default to localhost
const VERIFY_SERVER_URL =
  process.env.CLAWBIZARRE_VERIFY_URL || "http://localhost:8340";

const MAX_CODE_SIZE = 50_000; // 50KB safety limit
const SUPPORTED_LANGUAGES = ["python", "javascript", "bash"];
const REQUEST_TIMEOUT_MS = 30_000; // 30s timeout for verification

interface VerifyRequest {
  code: string;
  test_suite: string;
  language?: string;
  use_docker?: boolean;
}

// Required: execute verification against our verify_server
export async function executeJob(request: VerifyRequest): Promise<ExecuteJobResult> {
  const language = request.language || "python";
  const useDocker = request.use_docker || false;

  // Build verify_server payload
  const payload = {
    code: request.code,
    verification: {
      test_suite: {
        code: request.test_suite,
        language: language,
      },
      use_docker: useDocker,
    },
  };

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    const response = await fetch(`${VERIFY_SERVER_URL}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errorText = await response.text().catch(() => "Unknown error");
      return {
        deliverable: JSON.stringify({
          verification_result: "ERROR",
          error: `Verification server returned ${response.status}: ${errorText}`,
          verified_at: new Date().toISOString(),
        }),
      };
    }

    const receipt = await response.json();

    return {
      deliverable: JSON.stringify({
        verification_result: receipt.tier_0_passed ? "PASS" : "FAIL",
        tests_passed: receipt.test_results?.passed || 0,
        tests_failed: receipt.test_results?.failed || 0,
        tests_total:
          (receipt.test_results?.passed || 0) +
          (receipt.test_results?.failed || 0),
        language: language,
        docker_sandboxed: useDocker,
        vrf_receipt: receipt,
        verified_at: new Date().toISOString(),
      }),
    };
  } catch (error: any) {
    if (error.name === "AbortError") {
      return {
        deliverable: JSON.stringify({
          verification_result: "ERROR",
          error: "Verification timed out after 30 seconds",
          verified_at: new Date().toISOString(),
        }),
      };
    }

    return {
      deliverable: JSON.stringify({
        verification_result: "ERROR",
        error: `Verification server unreachable: ${error.message}`,
        verified_at: new Date().toISOString(),
      }),
    };
  }
}

// Optional: validate incoming requests before accepting
export function validateRequirements(request: any): ValidationResult {
  if (!request.code || typeof request.code !== "string" || request.code.trim().length === 0) {
    return { valid: false, reason: "code is required and must be a non-empty string" };
  }

  if (!request.test_suite || typeof request.test_suite !== "string" || request.test_suite.trim().length === 0) {
    return { valid: false, reason: "test_suite is required and must be a non-empty string" };
  }

  if (request.code.length > MAX_CODE_SIZE) {
    return {
      valid: false,
      reason: `code exceeds maximum size of ${MAX_CODE_SIZE} bytes (got ${request.code.length})`,
    };
  }

  if (request.test_suite.length > MAX_CODE_SIZE) {
    return {
      valid: false,
      reason: `test_suite exceeds maximum size of ${MAX_CODE_SIZE} bytes (got ${request.test_suite.length})`,
    };
  }

  if (request.language && !SUPPORTED_LANGUAGES.includes(request.language)) {
    return {
      valid: false,
      reason: `Unsupported language: ${request.language}. Supported: ${SUPPORTED_LANGUAGES.join(", ")}`,
    };
  }

  return { valid: true };
}

// Optional: custom payment request message
export function requestPayment(request: any): string {
  const lang = request.language || "python";
  const docker = request.use_docker ? " (Docker sandboxed)" : "";
  return `Structural code verification — ${lang}${docker}`;
}
