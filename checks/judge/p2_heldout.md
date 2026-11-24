# P2·흐름 면제 — 조정에 쓰지 않은 big1 val/test 확인 (정답 규칙은 scripts/p2_heldout.py 머리말, 실행 전 커밋)

> 채점 28 (정지 20 · 느린 흐름 8) · 제외 2 · 정답 수 **G2 20/28** · G(P1 만) 18/28

| 클립 | 무늬 | 마스크 | 속도 상한(주기/프레임) | 정답 | G | G2 | G2 근거 |
|---|---|---|---|---|---|---|---|
| val/val_pattern_hazard_000 | straight | perspective | 0.257 | 안전 | 안전 | 안전 | {'area_static': 0.0, 'area_change': 0.0, 'area_flow': 0.316, 'reason': ''} |
| val/val_pattern_hazard_001 | wavy | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.463, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_002 | straight | static | 0.351 | 안전 | 위험 ✗ | 위험 ✗ | {'area_static': 0.0, 'area_change': 0.511, 'area_flow': 0.012, 'reason': '대비 반전'} |
| val/val_pattern_hazard_003 | concentric | static | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.543, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_004 | wavy | static | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.576, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_005 | chirp | perspective | 0.183 | 안전 | 위험 ✗ | 안전 | {'area_static': 0.001, 'area_change': 0.0, 'area_flow': 0.509, 'reason': ''} |
| val/val_pattern_hazard_006 | straight | perspective | 0.042 | 안전 | 위험 ✗ | 안전 | {'area_static': 0.054, 'area_change': 0.0, 'area_flow': 0.444, 'reason': ''} |
| val/val_pattern_hazard_007 | spiral | perspective | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.711, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_008 | concentric | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.811, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_009 | chirp | anchored | 0.231 | 안전 | 위험 ✗ | 위험 ✗ | {'area_static': 0.005, 'area_change': 0.543, 'area_flow': 0.021, 'reason': '대비 반전'} |
| val/val_pattern_hazard_010 | checker | perspective | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.719, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_011 | checker | perspective | 0.258 | 안전 | 안전 | 안전 | {'area_static': 0.119, 'area_change': 0.157, 'area_flow': 0.128, 'reason': '대비 반전'} |
| val/val_pattern_hazard_012 | chirp | perspective | 0.0 | 위험 | 안전 ✗ | 안전 ✗ | None |
| val/val_pattern_hazard_013 | chirp | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.894, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_014 | straight | perspective | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.734, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_015 | radial | anchored | 0.0 | 위험 | 안전 ✗ | 안전 ✗ | None |
| val/val_pattern_hazard_016 | straight | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.682, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| val/val_pattern_hazard_017 | wavy | anchored | 0.0 | 위험 | 안전 ✗ | 안전 ✗ | {'area_static': 0.404, 'area_change': 0.012, 'area_flow': 0.001, 'reason': '점멸·방향 바뀜'} |
| val/val_pattern_hazard_018 | radial | anchored | 0.0 | 위험 | 안전 ✗ | 안전 ✗ | None |
| val/val_pattern_hazard_019 | concentric | perspective | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.777, 'area_change': 0.024, 'area_flow': 0.006, 'reason': '점멸·방향 바뀜'} |
| test/test_pattern_hazard_000 | straight | anchored | 0.495 | - | 위험 | 위험 | {'area_static': 0.0, 'area_change': 0.702, 'area_flow': 0.046, 'reason': '진동·방향 전환'} |
| test/test_pattern_hazard_001 | concentric | anchored | 0.366 | 안전 | 위험 ✗ | 위험 ✗ | {'area_static': 0.0, 'area_change': 0.451, 'area_flow': 0.244, 'reason': '대비 반전'} |
| test/test_pattern_hazard_002 | chirp | static | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.682, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| test/test_pattern_hazard_003 | chirp | static | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.879, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| test/test_pattern_hazard_004 | spiral | static | 0.515 | - | 위험 | 위험 | {'area_static': 0.0, 'area_change': 0.453, 'area_flow': 0.051, 'reason': '대비 반전'} |
| test/test_pattern_hazard_005 | wavy | perspective | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.687, 'area_change': 0.0, 'area_flow': 0.0, 'reason': ''} |
| test/test_pattern_hazard_006 | spiral | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.634, 'area_change': 0.001, 'area_flow': 0.0, 'reason': '대비 반전'} |
| test/test_pattern_hazard_007 | wavy | anchored | 0.0 | 위험 | 위험 | 위험 | {'area_static': 0.612, 'area_change': 0.008, 'area_flow': 0.003, 'reason': '점멸·방향 바뀜'} |
| test/test_pattern_hazard_008 | radial | anchored | 0.237 | 안전 | 안전 | 안전 | None |
| test/test_pattern_hazard_009 | checker | static | 0.0 | 위험 | 안전 ✗ | 안전 ✗ | {'area_static': 0.319, 'area_change': 0.008, 'area_flow': 0.007, 'reason': '대비 반전'} |
