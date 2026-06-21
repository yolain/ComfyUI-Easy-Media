import { horizontalToVerticalFov, normalizePanoramaView } from '@/lib/panorama-camera'
import type { MultiTrackPanoramaView } from '@/types/multitrack'
import { cn } from '@/lib/utils'

interface PanoramaImagePreviewProps {
  alt: string
  className?: string
  imageId: string
  imageUrl: string
  view: MultiTrackPanoramaView
}

export function PanoramaImagePreview({
  alt,
  className,
  imageId,
  imageUrl,
  view,
}: Readonly<PanoramaImagePreviewProps>) {
  const normalized = normalizePanoramaView(view)
  const verticalFov = horizontalToVerticalFov(normalized.hfov, 16 / 9)
  const imageWidth = 360 / normalized.hfov * 100
  const centerLeft = 50 - normalized.yaw / normalized.hfov * 100
  const centerTop = 50 + normalized.pitch / verticalFov * 100

  return (
    <div
      data-testid={`panorama-image-preview-${imageId}`}
      className={cn('relative aspect-video w-full overflow-hidden bg-black', className)}
    >
      {[-1, 0, 1].map((turn) => (
        <img
          key={turn}
          src={imageUrl}
          alt={turn === 0 ? alt : ''}
          aria-hidden={turn === 0 ? undefined : true}
          className="pointer-events-none absolute top-1/2 max-w-none -translate-x-1/2 -translate-y-1/2 select-none"
          style={{
            left: `${centerLeft + turn * imageWidth}%`,
            top: `${centerTop}%`,
            width: `${imageWidth}%`,
          }}
          draggable={false}
        />
      ))}
    </div>
  )
}
