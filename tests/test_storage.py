from src.documents.storage import SupabaseStorage, normalize_supabase_url


def test_normalize_supabase_url_accepts_project_origin():
    assert (
        normalize_supabase_url("https://project-ref.supabase.co")
        == "https://project-ref.supabase.co"
    )


def test_normalize_supabase_url_removes_rest_api_path_and_whitespace():
    assert (
        normalize_supabase_url(" https://project-ref.supabase.co/rest/v1 ")
        == "https://project-ref.supabase.co"
    )


def test_normalize_supabase_url_removes_storage_api_path():
    assert (
        normalize_supabase_url("https://project-ref.supabase.co/storage/v1")
        == "https://project-ref.supabase.co"
    )


def test_object_url_uses_storage_api_under_project_origin():
    storage = SupabaseStorage(
        url=normalize_supabase_url("https://project-ref.supabase.co/rest/v1"),
        service_role_key="secret",
        bucket="smartroute-documents",
    )

    assert (
        storage._object_url("test_user/Cover Letter Generator.txt")
        == "https://project-ref.supabase.co/storage/v1/object/"
        "smartroute-documents/test_user/Cover%20Letter%20Generator.txt"
    )
