/**
 * Centralized theme configuration for Euglena frontend.
 * All color values should be defined here and imported by components.
 */

export const theme = {
  colors: {
    background: {
      primary: 'bg-white',
      secondary: 'bg-gray-50',
      tertiary: 'bg-gray-100',
      card: 'bg-white',
      input: 'bg-white',
    },
    text: {
      primary: 'text-black',
      secondary: 'text-gray-700',
      tertiary: 'text-gray-600',
      muted: 'text-gray-500',
      placeholder: 'text-gray-400',
    },
    border: {
      primary: 'border-gray-200',
      secondary: 'border-gray-300',
      input: 'border-gray-300',
    },
    status: {
      completed: {
        bg: 'bg-green-50',
        border: 'border-green-200',
        text: 'text-green-800',
        icon: 'text-green-600',
        badge: 'bg-green-100 text-green-800',
      },
      error: {
        bg: 'bg-red-50',
        border: 'border-red-200',
        text: 'text-red-800',
        icon: 'text-red-600',
        badge: 'bg-red-100 text-red-800',
      },
      inProgress: {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        text: 'text-blue-800',
        icon: 'text-blue-600',
        badge: 'bg-blue-100 text-blue-800',
      },
      pending: {
        bg: 'bg-yellow-50',
        border: 'border-yellow-200',
        text: 'text-yellow-800',
        icon: 'text-yellow-600',
        badge: 'bg-yellow-100 text-yellow-800',
      },
      default: {
        bg: 'bg-gray-100',
        border: 'border-gray-300',
        text: 'text-gray-800',
        icon: 'text-gray-600',
        badge: 'bg-gray-100 text-gray-800',
      },
    },
    button: {
      primary: {
        bg: 'bg-black',
        text: 'text-white',
        border: 'border-gray-800',
        hover: 'hover:bg-gray-800',
      },
      secondary: {
        bg: 'bg-gray-200',
        text: 'text-black',
        border: 'border-gray-300',
        hover: 'hover:bg-gray-300',
      },
      danger: {
        bg: 'bg-red-600',
        text: 'text-white',
        border: 'border-red-700',
        hover: 'hover:bg-red-700',
      },
    },
    markdown: {
      code: {
        bg: 'bg-gray-200',
        text: 'text-black',
      },
      pre: {
        bg: 'bg-gray-200',
        text: 'text-black',
      },
    },
  },
  spacing: {
    card: {
      padding: 'p-4',
      rounded: 'rounded-xl',
      border: 'border-2',
    },
    input: {
      padding: 'p-3',
      rounded: 'rounded-xl',
    },
  },
} as const;
