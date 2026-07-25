using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace HearMeOutDesktopLauncher
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                string root = FindProjectRoot(AppDomain.CurrentDomain.BaseDirectory);
                string script = Path.Combine(root, "run_hearmeout_desktop.ps1");
                string arguments = "-NoProfile -ExecutionPolicy Bypass -File " + Quote(script);
                if (args.Length > 0)
                {
                    arguments += " " + string.Join(" ", args.Select(Quote));
                }

                var psi = new ProcessStartInfo
                {
                    FileName = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "WindowsPowerShell", "v1.0", "powershell.exe"),
                    Arguments = arguments,
                    WorkingDirectory = root,
                    UseShellExecute = true,
                    WindowStyle = ProcessWindowStyle.Minimized
                };

                Process.Start(psi);

                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "HearMeOut", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
        }

        private static string FindProjectRoot(string start)
        {
            var dir = new DirectoryInfo(start);
            for (int i = 0; i < 6 && dir != null; i++, dir = dir.Parent)
            {
                if (File.Exists(Path.Combine(dir.FullName, "run_hearmeout_desktop.ps1")))
                {
                    return dir.FullName;
                }
            }

            throw new FileNotFoundException("Could not find run_hearmeout_desktop.ps1 near the launcher.");
        }

        private static string Quote(string value)
        {
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }
    }
}
