// Unity Editor utility that uploads the built Addressables output for a given
// platform to the Node.js (NestJS) server using HTTP Basic auth.
//
// Where to put it:
//   Assets/Editor/AddressablesUploader.cs   (must be under an "Editor" folder)
//
// Usage:
//   1. Build your Addressables content (Window > Asset Management > Addressables
//      > Groups > Build > New Build > Default Build Script). The output goes to
//      ServerData/<BuildTarget>/ by default.
//   2. Menu:  Tools > Addressables > Upload To Server
//
// The server URL/credentials below should match your deployment.

#if UNITY_EDITOR
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

public static class AddressablesUploader
{
    // ---- Configure these for your server ----
    private const string ServerUrl = "http://localhost:3001";
    private const string ApiUser = "unity";
    private const string ApiPass = "changeme";
    // -----------------------------------------

    [MenuItem("Tools/Addressables/Upload To Server")]
    public static void UploadCurrentPlatform()
    {
        string platform = EditorUserBuildSettings.activeBuildTarget.ToString();

        // Default Addressables remote build path: ServerData/<BuildTarget>
        string buildPath = Path.Combine(
            Directory.GetParent(Application.dataPath).FullName,
            "ServerData",
            platform);

        if (!Directory.Exists(buildPath))
        {
            EditorUtility.DisplayDialog(
                "Addressables Uploader",
                $"Build output not found:\n{buildPath}\n\n" +
                "Build your Addressables content first.",
                "OK");
            return;
        }

        string[] files = Directory.GetFiles(buildPath, "*", SearchOption.AllDirectories);
        if (files.Length == 0)
        {
            Debug.LogWarning("[AddressablesUploader] No files to upload.");
            return;
        }

        Debug.Log($"[AddressablesUploader] Uploading {files.Length} files for {platform}...");
        UploadFiles(platform, files);
    }

    private static void UploadFiles(string platform, string[] files)
    {
        var form = new WWWForm();
        foreach (string file in files)
        {
            byte[] data = File.ReadAllBytes(file);
            // The server stores by the (sanitized) original file name.
            form.AddBinaryData("files", data, Path.GetFileName(file));
        }

        string url = $"{ServerUrl}/api/bundles/{platform}";
        using UnityWebRequest req = UnityWebRequest.Post(url, form);

        string token = System.Convert.ToBase64String(
            Encoding.UTF8.GetBytes($"{ApiUser}:{ApiPass}"));
        req.SetRequestHeader("Authorization", $"Basic {token}");

        // Editor-blocking send (simple). For large uploads consider an async flow.
        var op = req.SendWebRequest();
        while (!op.isDone)
        {
            EditorUtility.DisplayProgressBar(
                "Uploading Addressables",
                $"{platform} ({req.uploadProgress:P0})",
                req.uploadProgress);
        }
        EditorUtility.ClearProgressBar();

        if (req.result == UnityWebRequest.Result.Success)
        {
            Debug.Log($"[AddressablesUploader] Done: {req.downloadHandler.text}");
        }
        else
        {
            Debug.LogError(
                $"[AddressablesUploader] Failed ({req.responseCode}): {req.error}\n" +
                req.downloadHandler.text);
        }
    }
}
#endif
