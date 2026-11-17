from tests.conftest import register


def test_register_and_me(client):
    headers = register(client)
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["email"] == "a@example.com"


def test_password_is_hashed(client, db):
    register(client, password="Password1")
    stored = db.users.find_one({"email": "a@example.com"})
    assert stored["password_hash"] != "Password1"
    assert stored["password_hash"].startswith("$2")


def test_duplicate_email_rejected(client):
    register(client)
    r = client.post("/auth/register",
                    json={"email": "A@example.com", "password": "Password1", "full_name": "x"})
    assert r.status_code == 409


def test_weak_password_rejected(client):
    for pw in ("short1", "alllettersonly", "12345678"):
        r = client.post("/auth/register", json={"email": "w@example.com", "password": pw, "full_name": "W"})
        assert r.status_code == 422, pw


def test_login_success_and_failure(client):
    register(client)
    ok = client.post("/auth/login", json={"email": "a@example.com", "password": "Password1"})
    assert ok.status_code == 200 and "access_token" in ok.json()
    bad = client.post("/auth/login", json={"email": "a@example.com", "password": "Wrong123"})
    assert bad.status_code == 401
    unknown = client.post("/auth/login", json={"email": "nobody@example.com", "password": "Password1"})
    assert unknown.status_code == 401
    assert bad.json() == unknown.json()


def test_protected_route_requires_token(client):
    assert client.get("/papers").status_code == 401
    assert client.get("/papers", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_refresh_token_cannot_be_used_as_access(client):
    r = client.post("/auth/register", json={"email": "c@example.com", "password": "Password1", "full_name": "C"})
    refresh = r.json()["refresh_token"]
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {refresh}"}).status_code == 401
    new = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert new.status_code == 200
    assert client.post("/auth/refresh", json={"refresh_token": r.json()["access_token"]}).status_code == 401
