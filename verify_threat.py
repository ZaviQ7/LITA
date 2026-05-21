from engine import ThreatHUD
import config

def test_threat_scenario(hud, detections, expected_score, expected_status):
    score, status, color = hud.calculate_score(detections)
    print(f"Scenario detections: {[det['label'] for det in detections]}")
    print(f" -> Computed Score: {score} (Expected: {expected_score})")
    print(f" -> Computed Status: {status} (Expected: {expected_status})")
    assert score == expected_score, f"Expected score {expected_score}, got {score}"
    assert status == expected_status, f"Expected status {expected_status}, got {status}"
    print(" -> SUCCESS")

def main():
    print("==================================================")
    print("      LITA HEURISTIC THREAT SCORING VERIFIER      ")
    print("==================================================")
    
    hud = ThreatHUD()
    
    print("\n[verify_threat] Running Scenario 1: No detections...")
    test_threat_scenario(hud, [], 0, "SAFE")
    
    print("\n[verify_threat] Running Scenario 2: Known Person only...")
    test_threat_scenario(hud, [
        {"class": "person", "is_known": True, "label": "KNOWN: Jane"}
    ], 0, "SAFE")
    
    print("\n[verify_threat] Running Scenario 3: Unknown Person only...")
    test_threat_scenario(hud, [
        {"class": "person", "is_known": False, "label": "UNKNOWN"}
    ], 30, "ELEVATED")
    
    print("\n[verify_threat] Running Scenario 4: Weapon (knife) only...")
    test_threat_scenario(hud, [
        {"class": "knife", "label": "WEAPON (KNIFE)"}
    ], 60, "CRITICAL")
    
    print("\n[verify_threat] Running Scenario 5: Unknown Person + Weapon...")
    test_threat_scenario(hud, [
        {"class": "person", "is_known": False, "label": "UNKNOWN"},
        {"class": "scissors", "label": "WEAPON (SCISSORS)"}
    ], 90, "CRITICAL")
    
    print("\n[verify_threat] Running Scenario 6: Multiple weapons + Unknown Person...")
    test_threat_scenario(hud, [
        {"class": "person", "is_known": False, "label": "UNKNOWN"},
        {"class": "knife", "label": "WEAPON (KNIFE)"},
        {"class": "scissors", "label": "WEAPON (SCISSORS)"}
    ], 90, "CRITICAL")
    
    print("\n==================================================")
    print(" SUCCESS: LITA HEURISTIC DECISION ENGINE VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    main()
