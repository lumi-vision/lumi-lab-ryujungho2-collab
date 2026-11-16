using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using System.Collections.Generic;

internal static class LumiLauncher
{
    static readonly string Root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
    static int Port = 8765;
    static string Url { get { return "http://127.0.0.1:" + Port + "/"; } }
    static string Log { get { return Path.Combine(Root, "data", "launcher", "server-" + Port + ".log"); } }

    [STAThread]
    static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        try
        {
            for (int i = 0; i < args.Length; i++)
                if (args[i] == "--port" && i + 1 < args.Length) Port = int.Parse(args[++i]);
            if (Port < 1024 || Port > 65535) throw new Exception("포트 범위: 1024~65535");
            if (Array.IndexOf(args, "--check") >= 0)
            {
                EnsureServer();
                CheckPage("generate", "GAN");
                CheckPage("", "LUMI");
                Directory.CreateDirectory(Path.Combine(Root, "data", "launcher"));
                File.WriteAllText(Path.Combine(Root, "data", "launcher", "check-" + Port + ".txt"),
                    "PASS\r\nRoot: " + Root + "\r\nURL: " + Url + "\r\n" + DateTime.Now.ToString("s"));
                return 0;
            }
            string name = Path.GetFileNameWithoutExtension(Application.ExecutablePath);
            string mode = name.Contains("생성기") || Array.IndexOf(args, "--generate") >= 0 ? "generate" :
                          name.Contains("플레이어") || Array.IndexOf(args, "--player") >= 0 ? "player" : null;
            if (mode != null)
            {
                EnsureServer();
                OpenBrowser(mode);
                return 0;
            }
            ShowMenu();
            return 0;
        }
        catch (Exception ex)
        {
            Directory.CreateDirectory(Path.Combine(Root, "data", "launcher"));
            File.WriteAllText(Path.Combine(Root, "data", "launcher", "last-error.txt"), ex.ToString());
            if (Array.IndexOf(args, "--check") < 0)
                MessageBox.Show(ex.Message + "\r\n\r\n로그: " + Log, "LUMI 실행 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    static string Read(string path)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(Url + path);
        request.Proxy = null;
        request.Timeout = 2000;
        using (WebResponse response = request.GetResponse())
        using (StreamReader reader = new StreamReader(response.GetResponseStream())) return reader.ReadToEnd();
    }

    static bool Healthy()
    {
        try
        {
            Dictionary<string, object> h = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(Read("api/health"));
            return (string)h["app"] == "lumi-local" &&
                String.Equals(Path.GetFullPath((string)h["root"]).TrimEnd(Path.DirectorySeparatorChar), Root, StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    static void EnsureServer()
    {
        if (Healthy()) return;
        string python = Path.Combine(Root, ".venv", "Scripts", "python.exe");
        string script = Path.Combine(Root, "scripts", "launch_server.py");
        if (!File.Exists(python) || !File.Exists(script))
            throw new Exception("이 실행 파일은 LUMI 프로젝트 폴더에서 실행해야 합니다.\r\n.venv 또는 scripts 폴더가 없습니다. EXE만 다른 폴더로 옮기지 마세요.");
        using (TcpClient client = new TcpClient())
        {
            try { client.Connect("127.0.0.1", Port); }
            catch (SocketException) { }
            if (client.Connected)
                throw new Exception("포트 " + Port + "에 다른 서버 또는 이전 버전이 실행 중입니다. 해당 서버를 종료한 뒤 다시 실행하세요.");
        }
        ProcessStartInfo info = new ProcessStartInfo(python, "\"" + script + "\" --port " + Port);
        info.WorkingDirectory = Root;
        info.UseShellExecute = false;
        info.CreateNoWindow = true;
        info.WindowStyle = ProcessWindowStyle.Hidden;
        using (Process bootstrap = Process.Start(info))
        {
            for (int n = 0; n < 120; n++)
            {
                if (Healthy()) return;
                if (bootstrap.HasExited && bootstrap.ExitCode != 0)
                    throw new Exception("서버 시작에 실패했습니다. Python 환경과 서버 로그를 확인하세요.");
                Thread.Sleep(250);
            }
        }
        throw new Exception("서버가 준비되지 않았습니다. 서버 로그를 확인하세요.");
    }

    static void CheckPage(string path, string expected)
    {
        if (!Read(path).Contains(expected)) throw new Exception("화면 응답 확인 실패: " + path);
    }

    static void OpenBrowser(string mode)
    {
        Process.Start(new ProcessStartInfo(Url + (mode == "generate" ? "generate" : "")) { UseShellExecute = true });
    }

    static void ShowMenu()
    {
        Form form = new Form { Text = "LUMI", ClientSize = new Size(520, 280), StartPosition = FormStartPosition.CenterScreen,
                               FormBorderStyle = FormBorderStyle.FixedDialog, MaximizeBox = false, Font = new Font("Malgun Gothic", 10) };
        Label title = new Label { Text = "LUMI 영상 합성 · 검사", Location = new Point(28, 24), AutoSize = true, Font = new Font("Malgun Gothic", 17, FontStyle.Bold) };
        Label status = new Label { Text = "작업을 선택하면 로컬 서버를 확인하고 브라우저를 엽니다.", Location = new Point(28, 190), Size = new Size(465, 60) };
        Button generate = new Button { Text = "1. GAN 영상 생성", Location = new Point(28, 90), Size = new Size(220, 70) };
        Button inspect = new Button { Text = "2. 탐지·검사 플레이어", Location = new Point(266, 90), Size = new Size(225, 70) };
        Func<string, Task> open = async mode =>
        {
            generate.Enabled = inspect.Enabled = false;
            status.Text = "서버를 확인하고 있습니다…";
            try { await Task.Run(() => EnsureServer()); OpenBrowser(mode); status.Text = "실행했습니다. 이 창을 닫아도 로컬 서버는 유지됩니다."; }
            catch (Exception ex) { status.Text = ex.Message; MessageBox.Show(form, ex.Message + "\r\n로그: " + Log, "실행 실패"); }
            finally { generate.Enabled = inspect.Enabled = true; }
        };
        generate.Click += async (s, e) => await open("generate");
        inspect.Click += async (s, e) => await open("player");
        form.Controls.AddRange(new Control[] { title, generate, inspect, status });
        Application.Run(form);
    }
}
