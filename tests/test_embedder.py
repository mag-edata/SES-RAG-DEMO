"""
Unit tests for embedder.py
Test cases: E-01, E-02, E-05
"""
import pytest

from rag.embedder import build_profile_text, load_engineers


# E-01: Load engineers.json and return a list of 20 records
def test_load_engineers_returns_20_records():
    engineers = load_engineers()
    assert isinstance(engineers, list)
    assert len(engineers) == 20


# E-02: build_profile_text returns a string containing name, skills, and years of experience
def test_build_profile_text_contains_required_fields():
    engineer = {
        "id": "eng_001",
        "氏名": "eng_001",
        "年齢": 32,
        "経験年数": 8.5,
        "所属": "プロパー",
        "参画可能時期": "即日",
        "概要": "Javaをメインとしたバックエンド開発者。金融・保険系の大規模システム開発経験が豊富。Spring Bootによるマイクロサービス設計・実装を得意とし、既存レガシーシステムのモダナイズ案件も多数経験。",
        "スキル": ["Java", "Spring Boot", "Oracle DB", "Oracle SQL", "REST API", "Git", "Maven", "JUnit", "マイクロサービス", "Linux"],
        "資格": ["Oracle Java SE 11 Silver", "AWS Solutions Architect Associate"],
        "勤務形態": "常駐・リモート両可",
        "希望勤務地": "東京・神奈川",
        "経歴": [
            "大手保険会社の保険金計算システムリプレイス（Java/Spring Boot、3年）",
            "証券会社の注文管理システムAPI開発（Java/REST API、2年）",
            "銀行の顧客管理システム設計・実装（Java/Oracle、1年）",
        ],
        "現在単価": "75万円/月",
    }
    text = build_profile_text(engineer)
    assert isinstance(text, str)
    assert len(text) > 0
    assert "eng_001" in text
    assert "Java" in text
    assert "8.5" in text


# E-05: Passing a non-existent path raises FileNotFoundError
def test_load_engineers_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_engineers("/nonexistent/path/engineers.json")
