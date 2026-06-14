// Runtime helper: attach HTTP Basic auth credentials to every Addressables
// download request. Only needed if the server is started with
// PROTECT_DOWNLOADS=true (otherwise the /content path is public).
//
// Where to put it:
//   Assets/Scripts/AddressablesAuth.cs
//
// It runs automatically before the first scene loads and registers a
// WebRequestOverride on the Addressables system.

using System.Text;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.Networking;

public static class AddressablesAuth
{
    // Must match the server credentials (API_USER / API_PASS).
    private const string ApiUser = "unity";
    private const string ApiPass = "changeme";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Init()
    {
        string token = System.Convert.ToBase64String(
            Encoding.UTF8.GetBytes($"{ApiUser}:{ApiPass}"));

        Addressables.WebRequestOverride = (UnityWebRequest request) =>
        {
            request.SetRequestHeader("Authorization", $"Basic {token}");
        };
    }
}
