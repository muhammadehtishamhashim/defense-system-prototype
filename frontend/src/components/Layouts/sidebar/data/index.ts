import * as Icons from "../icons";

export const NAV_DATA = [
  {
    label: "DEFENSE SYSTEM V3",
    items: [
      {
        title: "Command Center",
        url: "/",
        icon: Icons.HomeIcon,
        items: [],
      },
      {
        title: "Monitoring",
        icon: Icons.PieChart,
        items: [
          {
            title: "Threat Detection",
            url: "/monitor/threat",
          },
          {
            title: "Theft Detection",
            url: "/monitor/theft",
          },
          {
            title: "Border Control",
            url: "/monitor/border",
          },
        ],
      },
      {
        title: "Verification",
        url: "/verify",
        icon: Icons.Table,
        items: [],
      },
      {
        title: "Settings",
        url: "/pages/settings",
        icon: Icons.Alphabet,
        items: [],
      },
    ],
  },
];
